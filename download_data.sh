#/bin/bash
# This has to be done with sudo command

wget -O data_db23.zip http://datadryad.org/api/v2/datasets/doi%253A10.5061%252Fdryad.1k84r/download
unzip data_db23.zip

mkdir txt
mv *.txt txt/

mkdir db2 db3

for i in {1..40}
do
    mkdir db2/s$i
    mv DB2_s$i.zip db2/s$i/DB2_s$i.zip

    cd db2/s$i
    unzip *.zip
    cd ../..
done

for i in {1..11}
do
    mkdir db3/s$i
    mv DB3_s$i.zip db3/s$i/DB3_s$i.zip

    cd db3/s$i
    unzip *.zip
    cd ../..
done

rm *.zip
