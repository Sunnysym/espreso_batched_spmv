#!/usr/bin/bash

matrixnum=$1
imbalanceratio=$2

echo ${matrixnum}
echo ${imbalanceratio}

for ((a=0; a<=6; a+=1));do
{
  k=a*100
  for((i=k+1; i<=k+100; i+=1));do
  {
    j=`printf %03d $i`
    echo $j
    read_dual_and_balance ${matrixnum} ${imbalanceratio} > read_dual_${matrixnum}.log
  }&
  done
  sleep 10
}
done

for ((i=701; i<= 712; i += 1));do
{
    j=`printf %03d $i`
    echo $j
    read_dual_and_balance ${matrixnum} ${imbalanceratio} > read_dual_${matrixnum}.log
}&
done

#for ((i=100; i<= 101; i += 1));do
#{
#    j=`printf %03d $i`
#    ln  /work1/wangjue/linkehao/5.modify_geo/newgeo/vtk1/rlzj.VOLUME${j}.vtk 032/rlzj.VOLUME${j}.vtk
#
#    ln  /work1/wangjue/linkehao/5.modify_geo/newgeo/vtk1/rlzj.TOP${j}.vtk 032/rlzj.TOP${j}.vtk
#
#    ln  /work1/wangjue/linkehao/5.modify_geo/newgeo/vtk1/rlzj.BOTTOM${j}.vtk 032/rlzj.BOTTOM${j}.vtk
#
#}&
#done