cd ../../dataset2metadata/text_detection

echo $(pwd)

start_id=$1
#end_id = start_id + 10
end_id=$((start_id+9))

for i in $(seq $start_id $end_id) 
do
    file_name=/project_data/datasets/laion400m-met-release/all_templates/$i.yml
    echo $file_name
    dataset2metadata --yml $file_name
done
