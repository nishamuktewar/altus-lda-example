# altus-lda-example
## Build
```
cd [wherever the project directory is]
mvn -Pspark-deploy package
```
## Deploy
````
spark2-submit \
 --master yarn --deploy-mode client \
 --executor-memory 10g \
 --driver-memory 10g \
 --conf spark.yarn.executor.memoryOverhead=5g \
 --conf spark.driver.extraJavaOptions=-Dbigdl.check.singleton=false \
 --conf spark.shuffle.reduceLocality.enabled=false \
 --conf spark.driver.maxResultSize=5g \
 --num-executors 4 --executor-cores 4 \
 --conf spark.dynamicAllocation.enabled=false \
 --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
 --conf spark.rpc.message.maxSize=800 \
 --conf "spark.kryoserializer.buffer.max=2047m" \
 --class "com.cloudera.datascience.LDAExample" \
 altus-lda-example-1.0-SNAPSHOT-jar-with-dependencies.jar \
 	--dataDir hdfs:///user/sowen/DataSets/gutenberg \
 	--stopwordFile hdfs:///user/nisha/stopwords.txt \
 	--saveModelDir hdfs:///user/nisha/LDAModels
````
