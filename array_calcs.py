from numpy import *

def get_roots(f_x,x,x_lim,tol,*args,**kwargs):
  f_ans = f_x(x,*args,**kwargs)
  if ndim(f_ans) == 1:
    f = f_ans; 
    tiny = 1.0e-5; df = (f_x(x*(1.0+tiny),*args,**kwargs) - f)/(x*tiny)
  else: f = f_ans[0]; df = f_ans[1]
  if any(x):
    x_diff = median(append(0.75*(x_lim-array([x,x])),[-f/df],0),0)
    x += x_diff
    ifx = (absolute(x_diff) > tol) & (absolute(f) > tol)
    x[ifx],f[ifx],df[ifx] = get_roots(f_x,x[ifx],x_lim[:,ifx],tol,
                                      *args,**kwargs)
  return (x,f,df)

def integrate_unsorted(f_x,x,x_0=0.0,min_sub=100,*args,**kw_args):
  i_srt = argsort(x)
  F_array = integrate_sorted(f_x,x[i_srt],x_0,min_sub,*args,**kw_args)
  return F_array[argsort(i_srt)]

def integrate_sorted(f_x,x,x_0=0.0,min_sub=100,*args,**kw_args):
  if any(x):
    len_x = len(x)
    if x[0] <= x_0 <= x[-1]:
      xl = x[x < x_0]; min_subl = ceil(min_sub*len(xl)/len_x)
      xr = x[x > x_0]; min_subr = ceil(min_sub*len(xr)/len_x)
      Fl_array = integrate_sorted(f_x,xl,x_0,min_subl,*args,**kw_args)
      Fr_array = integrate_sorted(f_x,xr,x_0,min_subr,*args,**kw_args)
      return hstack((Fl_array,0.0*x[x==x_0],Fr_array)).ravel()
    else:
      i_arr = arange(len_x) + (x_0 > x[-1])*arange(len_x-1,-len_x,-2)
      x = append(x_0,x[i_arr]); f = f_x(x,*args,**kw_args)
      F_array = empty(len_x); max_dx = (x[-1] - x_0)/min_sub; F = 0.0
      for dx,x,f_ave,i in zip(x[1:]-x[:-1],x,0.5*(f[1:] + f[:-1]),i_arr):
        num_traps = ceil(dx/max_dx)
        if num_traps > 1:
          x_int,dx = linspace(x,x+dx,num_traps+1,retstep=True)
          F += dx*(f_ave + sum(f_x(x_int[1:-1],*args,**kw_args)))
        else: F += dx*f_ave
        F_array[i] = F
      return F_array
