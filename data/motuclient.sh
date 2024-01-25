#!/bin/bash
conda activate py39

python3 -m site
#python3 -m motuclient
#echo "hello"
#. activate py39
#python -m motuclient --motu https://my.cmems-du.eu/motu-web/Motu \
#--service-id BLKSEA_MULTIYEAR_PHY_007_004-TDS --product-id {} \
#--longitude-min 27.37 --longitude-max 41.9626 --latitude-min 40.86 --latitude-max 46.8044 \
#--date-min "1993-01-01 00:00:00" --date-max "2021-06-30 23:59:59" --depth-min 12.5 \
#--depth-max 12.536195755004883 --variable uo --variable vo \
#--out-dir /mnt/miyuan/AI4Physics/Data/bs --out-name {target}.nc --user ymi --pwd My12345678'