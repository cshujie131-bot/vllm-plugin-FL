"""FlagGems-compatible in-place mask; caches launch metadata, never tensor data/pointers."""
import torch,triton
import triton.language as tl
from functools import lru_cache
from flag_gems.ops.masked_fill import masked_fill_ as generic
_CACHES={}
_VERSION=tuple(int(x) for x in triton.__version__.split('.')[:2])

@triton.jit(do_not_specialize=['value'])
def _masked_fill_store(X,M,value,N:tl.constexpr,C:tl.constexpr,
                       X0:tl.constexpr,X1:tl.constexpr,M0:tl.constexpr,M1:tl.constexpr,
                       B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B)
 row=i//C;col=i%C
 m=tl.load(M+row*M0+col*M1,i<N,other=0).to(tl.int1)
 x=tl.load(X+row*X0+col*X1,i<N,other=0)
 out=tl.where(m,value.to(x.dtype),x)
 tl.store(X+row*X0+col*X1,out,i<N)

@triton.jit(do_not_specialize=['value'])
def _masked_fill_rows(X,M,value,C:tl.constexpr,X0:tl.constexpr,M0:tl.constexpr,B:tl.constexpr,LOOP:tl.constexpr):
 row=tl.program_id(0);cols=tl.program_id(1)*B+tl.arange(0,B)
 flag=tl.load(M+row*M0).to(tl.int1)
 if flag:
  if LOOP:
   for base in range(tl.cdiv(C,B)):
    col=base*B+tl.arange(0,B)
    tl.store(X+row*X0+col,value,col<C)
  else:tl.store(X+row*X0+cols,value,cols<C)

@lru_cache(256)
def _layout(xs,xt,ms,mt):
 if len(xs)==0:return 1,1,0,0,0,0
 if len(xs)>2 or len(ms)>len(xs):return None
 if len(xs)==1:xs=(1,)+xs;xt=(0,)+xt
 ms=(1,)*(2-len(ms))+ms;mt=(0,)*(2-len(mt))+mt
 if any(m!=1 and m!=x for x,m in zip(xs,ms)):return None
 mt=tuple(0 if m==1 else s for m,s in zip(ms,mt))
 return xs[0],xs[1],xt[0],xt[1],mt[0],mt[1]

def masked_fill_(inp,mask,value):
 if mask.dtype!=torch.bool:raise RuntimeError('masked_fill_ requires boolean mask')
 if not isinstance(value,(float,int,bool)) or inp.dtype not in (torch.float32,torch.bfloat16,torch.float16,torch.int32,torch.int64,torch.bool):
  return generic(inp,mask,value)
 meta=_layout(tuple(inp.shape),tuple(inp.stride()),tuple(mask.shape),tuple(mask.stride()))
 if meta is None:return generic(inp,mask,value)
 rows,cols,x0,x1,m0,m1=meta;n=rows*cols
 if not n:return inp
 # Overlapping expanded output is not supported; retain generic semantics.
 if (rows>1 and x0==0) or (cols>1 and x1==0):return generic(inp,mask,value)
 rowpath=bool(m1==0 and x1==1 and cols>=512 and rows>=16)
 B=(1024 if rowpath else 4096) if n>=262144 else 256
 key=(rowpath,inp.device.index,inp.dtype,type(value),meta,B,inp.data_ptr()%16,mask.data_ptr()%16)
 if rowpath:
  fn=_masked_fill_rows;args=(inp,mask,value,cols,x0,m0,B,rows>=512);grid=(rows,1 if rows>=512 else triton.cdiv(cols,B))
 else:
  fn=_masked_fill_store;args=(inp,mask,value,n,cols,x0,x1,m0,m1,B);grid=(triton.cdiv(n,B),)
 kernel=_CACHES.get(key)
 if kernel is None:
  kernel=fn[grid](*args,num_warps=4 if B>=1024 else 4,num_stages=1)
  _CACHES[key]=kernel
 else:
  # Matches this installation's FlagGems LibEntry compiled-launch ABI.
  kernel[(grid+(1,1))[:3]](*(args if _VERSION[0]==3 and 3<=_VERSION[1]<=6 else args[:3]))
 return inp
