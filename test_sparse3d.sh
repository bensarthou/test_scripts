#!/bin/sh

# for t in 1 2 3
# do
#   for T in 1 2 3 4 5 6 7 8 9 10 11 12 13 14
#   do
#       for l in 1 2 3 4 5 6 7
#       do
#
#           mr3d_trans -t $t -n 4 ~/src/baboun_ref_128x128x128_ref_MID14_FID24.fits ~/src/sparse3d_res/output_trans_${t}
#           mr3d_trans -t $t -n 4 -l $l ~/src/baboun_ref_128x128x128_ref_MID14_FID24.fits  ~/src/sparse3d_res/output_trans_${t}_${l}
#           mr3d_trans -t $t -n 4 -L ~/src/baboun_ref_128x128x128_ref_MID14_FID24.fits  ~/src/sparse3d_res/output_trans_${t}_L
#           mr3d_trans -t $t -n 4 -l $l -L ~/src/baboun_ref_128x128x128_ref_MID14_FID24.fits  ~/src/sparse3d_res/output_trans_${t}_${l}_L
#
#           mr3d_trans -t $t -n 4 -T $T ~/src/baboun_ref_128x128x128_ref_MID14_FID24.fits ~/src/sparse3d_res/output_trans_${t}_${T}
#           mr3d_trans -t $t -n 4 -T $T -l $l ~/src/baboun_ref_128x128x128_ref_MID14_FID24.fits  ~/src/sparse3d_res/output_trans_${t}_${T}_${l}
#           mr3d_trans -t $t -n 4 -T $T -L ~/src/baboun_ref_128x128x128_ref_MID14_FID24.fits  ~/src/sparse3d_res/output_trans_${t}_${T}_L
#           mr3d_trans -t $t -n 4 -T $T -l $l -L ~/src/baboun_ref_128x128x128_ref_MID14_FID24.fits  ~/src/sparse3d_res/output_trans_${t}_${T}_${l}_L
#
#       done
#   done
# done


for t in 1 2
do
  for T in 1 2 3 4 5 6 7 8 9 10 11 12 13 14
  do
      mr3d_filter -t $t -n 4 ~/src/baboun_ref_128x128x128_ref_MID14_FID24.fits ~/src/sparse3d_res/output_filter_${t}
      mr3d_filter -t $t -n 4 -C ~/src/baboun_ref_128x128x128_ref_MID14_FID24.fits ~/src/sparse3d_res/output_filter_${t}_C
      mr3d_filter -t $t -n 4 -g 3 ~/src/baboun_ref_128x128x128_ref_MID14_FID24.fits ~/src/sparse3d_res/output_filter_${t}_g
      mr3d_filter -t $t -n 4 -g 3-C ~/src/baboun_ref_128x128x128_ref_MID14_FID24.fits ~/src/sparse3d_res/output_filter_${t}_C_g

      mr3d_filter -t $t -n 4 -s 4 ~/src/baboun_ref_128x128x128_ref_MID14_FID24.fits ~/src/sparse3d_res/output_filter_${t}_s
      mr3d_filter -t $t -n 4 -s 4 -C ~/src/baboun_ref_128x128x128_ref_MID14_FID24.fits ~/src/sparse3d_res/output_filter_${t}_C_s
      mr3d_filter -t $t -n 4 -s 4 -g 3 ~/src/baboun_ref_128x128x128_ref_MID14_FID24.fits ~/src/sparse3d_res/output_filter_${t}_g_s
      mr3d_filter -t $t -n 4 -s 4 -g 3 -C ~/src/baboun_ref_128x128x128_ref_MID14_FID24.fits ~/src/sparse3d_res/output_filter_${t}_C_g_s

      mr3d_filter -t $t -T $T -n 4 ~/src/baboun_ref_128x128x128_ref_MID14_FID24.fits ~/src/sparse3d_res/output_filter_${t}_${T}
      mr3d_filter -t $t -T $T -n 4 -C ~/src/baboun_ref_128x128x128_ref_MID14_FID24.fits ~/src/sparse3d_res/output_filter_${t}_${T}_C
      mr3d_filter -t $t -T $T -n 4 -g 3 ~/src/baboun_ref_128x128x128_ref_MID14_FID24.fits ~/src/sparse3d_res/output_filter_${t}_${T}_g
      mr3d_filter -t $t -T $T -n 4 -g 3 -C ~/src/baboun_ref_128x128x128_ref_MID14_FID24.fits ~/src/sparse3d_res/output_filter_${t}_${T}_C_g

      mr3d_filter -t $t -T $T -n 4 -s 4 -C ~/src/baboun_ref_128x128x128_ref_MID14_FID24.fits ~/src/sparse3d_res/output_filter_${t}_${T}_C_s
      mr3d_filter -t $t -T $T -n 4 -s 4 ~/src/baboun_ref_128x128x128_ref_MID14_FID24.fits ~/src/sparse3d_res/output_filter_${t}_${T}_s
      mr3d_filter -t $t -T $T -n 4 -s 4 -g 3 ~/src/baboun_ref_128x128x128_ref_MID14_FID24.fits ~/src/sparse3d_res/output_filter_${t}_${T}_g_s
      mr3d_filter -t $t -T $T -n 4 -s 4 -g 3 -C ~/src/baboun_ref_128x128x128_ref_MID14_FID24.fits ~/src/sparse3d_res/output_filter_${t}_${T}_C_g_s

  done
done
