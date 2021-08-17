# DATA ENGINEERING

##### 28/07/2021

### AWS s3

1. Max file size 5TB
2. Allows any file type to store
3. Common ML data file formats - csv, json, protobuf, orc, avro, parquet
4. Data **partitioning** techniques - by date, by product, athena (serverless), Kinesis or Glue (does partinioning for us)
5. 5 types of s3 **creation**. The more we move towards the less cost it induces. We can also create rules for transitioning from one type of bucket to another. This can be done under Manage bucket section.
6. **Encryption** - 4 types of encrytion sse-s3, sse-kms, sse-c, client-side encyption. For Ml we mostly use sse-3 and sse-kms
7. **SSE-3** - (Server SIde Encryption-3) When we store object to s3, s3 generated a key and attaches it to our data.
8. **SSE-KMS** - (Server Side Encrytion Key Management Service CMK (Customer Master Key)) When we store object to s3, KMS-CMK creates encryption which we can manage from our end. This provides more security
9. For more security, define policies fro accessing bucket. 2 ways - User based - IAM policy (json), Resource based policy - bucket policy, object acl, bucket acl
10. bucket policy - very commonly used to grant cross account access
11. Other security topiccs -
    1. Security via - **VPC** Endpoint gateway - usually when vpc is not configured, all our data goes through public network to read/write s3
    2. If VPC configured, it creates a private network for such reads/writes.
    3. **Note** - check resources like sagemaker to access s3 in such a private network.
    4. Security via AWS event logging and audit is done by **Cloudtrail**
    5. Security via **Tagging** s3 object via properties with Classification=PHI (Personal Health Information). Onlu right bucket policy or IAM policy can access such objects.
12. **TODO - create a bucket with above features**

##### 29/07/2021 - 02/08/2021

### AWS Kinesis

1. Kinesis is a streaming servicce, an alternative to Apache Kafka
2. Compatible with stream processing framework like Spark, NiFi, etc
3. Data is by default replicated to 3 regions, so it is highly safe
4. **Exam Que** - Diff between - Streams, Analystics, Firehose, Video Streams
   1. Kinesis Stream

      1. realtime, low latency streaming ingest
      2. Streams are divided into Shards/Partitions
      3. Hards need to be provisioned in advance
      4. Data rention 24hrs by default, can be extended to 7 days
      5. Due to above point, it provides playback or replay feature; also multiple apps can use same stream
      6. Records can be upto 1MB size
      7. Data cannot be deleted once inserted to Kinesis
      8. Limitations -
         1. Producer - 1MB/s or 1000 messages/s write per shard is allowed, above that you get error 'ProvisionThroughputException'
         2. Consumer - 2MB/s read per shard or 5api calls/s per shard\
         3. Higher the no. of Shards, greater the capacity and speed. Hence KS only scales when we add shards over time
   2. Kinesis Analytics -

      1. realtime analytics (ETL - Extract,Transform,Load) of streams using SQL or Flink
      2. generate metrics in realtime, ex-games like ludo
      3. responsive analytics - realtime analysis based on certain criteria/input and filtering
      4. only pay for resource consumed (but it's not cheap)
      5. we can use IAM permissions
      6. serverless, scales automatically
      7. lambda can be used for data processing
      8. schema discovery
      9. **Machine Learning on KA**
         1. Random Cut Forest - anomaly detection on numeric columns, uses only recent data, hence model changes overtime. Ex- detect anomalies in webapp requests
         2. Hotspots - locates and returs information about dense region in our data. Less changing model. Ex- locate time in a day where max requests are made to an app
   3. Kinesis Firehose -

      1. Load data in s3, Redshift, ElasticSearch, Splunk
      2. Fully managed service, no need of administration
      3. Near realtime, 60sec latency minimum
      4. Automatic scaling
      5. Store data in any format
      6. Data Conversion and suppresssion allowed (zip, gzip etc) only for s3
      7. Data Transformation allowed only for lambda
   4. Diff bet Kinesis Streams and Kinesis Firehose

      1. KS latency 70-300ms, custom code for producer and consumer, real time applications, storage 1-7days, replay capability, multiple consumers, manual scaling (high throughput - shard splitting, less throughput - shard merging)
      2. KF is ingestion service, full managed, send data to (s3, redshift, elasticsearch, splunk), serverless data tansformation to lambda, near real-time, automated scaling, no data source hence no replay
   5. Kinesis Video Streams -

      1. streaming video real time
      2. one producer per video stream. producer example - security camera, body-worn camera, AWS deeplens, smartphone camera
      3. consumer example - AWS sagemaker, our own algorithm like mxnet or tf, AWS rekognition
      4. Data storage 1-10yrs
5. Architecture - KS takes all data from IOT devices, metric logs, etc OR KF stores them to Redshift, s3, etc for deeper analytics, reporting etc --> KA analyzes data stored by KS or KF
6. **Summary** - check kinesis.png

##### 29/07/2021 - 02/08/2021

### AWS Glue

1. **Glue Data Catalog**
   1. is a metadata repository for all the tables in your account
   2. Schemas are automatically created, schemas are versioned
   3. Integrated with athena, redshift, etc
2. **Glue crawlers**
   1. crawl through all your databases like dynamodb, s3, etc
   2. infers schemas and partitions
   3. works for json, parquet, csv, etc
   4. we can run crawler on demand or on schedule
   5. Based on how we will query s3 bucket, store data according beforehand.
3. **Glue ETL**
   1. Extract, Transform and Load, ETL allows you to clean data before analysis
   2. Target can be RDS, glue data catalog, s3
   3. Generate ETL code in python or spark
   4. Fully managed, cost effective, pay only for resources used
   5. Jobs are run on serverless spark platform
   6. Glue scheduler is used to schedule jobs
   7. Glue triggers are used to automate job runs based on event
   8. **Transformations** - Glue ETL is used for transforming data. Transformation can be applied via -
      1. **Bundled Transformations**
         1. DropFields, DropNullFields - dropping null fields
         2. Filter - filtering as per some criteria
         3. join - join data
         4. map - add or delete fields, perform external lookups
      2. **Machine Learnig Transformations**
         1. FindMatches ML - Find duplicate data even if the data are not perfectly matching
      3. Format conversions - csv, json, orc, parquet, xml
      4. Apache spark transformations (ex- k-means)

##### 05/08/2021

### AWS Athena

1. Querying tool for s3
2. Requirement - s3 data should be available in AWS glue in the form of tables

### AWS Datastores

1. Redshift -
   1. datawareshouse technology
   2. massively parallel sql queries to run, then use Redshift. (OLAP - Online Analytical Processing)
   3. load data from s3 OR query it from s3 using redshift spectrum
   4. provision servers in advance
   5. Data is organized in columns or tables
   6. main purpose is to do analytics
2. RDS and Aurora-
   1. These are relational datastores
   2. OLTP - Online Transaction Processing
   3. provision servers in advance
   4. Data is orgainized in row
   5. main purpose is to to store data
3. Dynamodb -
   1. NoSDL - NotOnlySQL (some systems stores sql plus document data) or Not SQL (not sql only document data is stores)
   2. Useful to store ml outputs
   3. serverless
   4. provisioned read/write capacity
4. s3 -
   1. object storage
   2. serverless
   3. infite storage
   4. integration with almost all aws services
5. elastic search -
   1. indexing of data
   2. provision in advance
   3. search among data points
   4. useful for analytics
6. elastic cache
   1. cahing mechanism

### AWS Data pipelines

1. ETL sesrvice
2. move data from one source to other
3. integration with Redshift, RDS, s3, Dynamodb, EMR
4. manage task dependency
5. retries and notifies on failure
6. highly available service

##### 06/08/2021

### AWS Batch

1. Runs job as docker images
2. Used for performing computational or etl or any kind of job in batches
3. No need of provisioning from user. Dynamically provisioned
4. All resources like ec2 are dynamically created as per the job
5. User does not have to manage clusters, it is serverless
6. Schedule batch jobs using cloudwatch or prchestrate batch jobs using AWS step functions

### AWS Database Migration Service

1. Used for migrating database safely
2. service runs on ec2 instance
3. It will just replicate the data from source to destination
4. Source database will remain accessible while the migration service is running and also afterwards
5. Homogenous migration - oracle to oracle
6. Heterogenous migration - microsoft server to aurora

### AWS Step Function

1. Used for visualizing all the steps to perform inyour application
2. It will show you flowchart with knowledge of which aws service to use at which point in building your ML application
3. Used to design workflows
4. advanced error handling and retry mechanism out of the code
5. audit of the history of workflows
6. ability to wait for an arbitary amount of time
7. Max execution time of state machine is 1 year

# EXPLORATORY DATA ANALYSIS

##### 07/08/2021

### Pandas

1. A library to load, analyze and help clean the data
2. Deals in dataframes(2D like table) and series(1D like array)
3. Ex - **df['country'].value_counts()** - gives name of country and it's total count in the data
4. Ex - df = pd.read_csv('temp.txt', na_values=[''])
5. Ex- **df.describe()**
6. Interoperates with numpy. Mostly the data is given in terms of numpy arrays to train or predict. Ex - **df['query'].values() return numpy query data**

### Matplotlib

1. Used for showcasing the outputs in charts and graphs
2. **Seaborn** lib is built over matplotlib, better to use now a days
3. Heatmap, Jointmap

### Scikit learn

### Jupyter Notebook

### Keras

1. Keras has a way to convert keras model into scikitlearn model and then train the data
2. This is done so that we can use scikitlearns cross_val_score()
   1. cross_val_score() randomly separates our data into train and validation
   2. Above step is done multiple times with param cv.

### Types of Data

1. **Numerical**
   1. numbers or integers like age, stock price, no. of requests on webpage
   2. Types -
      1. **Discrete** - no. of purchases made by me in a year, no. of attacks I got on my website this month
      2. **Continuous** - rainfall
2. **Categorical**
   1. yes or no, fruit classes
   2. Often one class has no corelation with other class. Like we cannot say apply is better/sweeter than orange. They are not comparable in that sense.
   3. Often this is converted into numerical in machine learning applications
3. **Ordinal**
   1. Mixture of of categorical and numerical
   2. Ex - star ratings (1-5 are numerical categories), also these numbers have corelation. Like 5 is better than 4.

### Data Distributions

1. Normal Distribution - continuous data
2. poisson distribution has Probability Mass Funciton - discrete data
3. Binomial Distribution - discrete data
   1. Distrbute data in such a way where the results is from only 2 case
   2. Ex - coin toss
4. Bernouli Distribution - discrete data
   1. Special case of Binomial distribution, consists of single trial
   2. Ex - coint toss only once
   3. Hence Binomial distribution can be considered as the collection of Bernouli distribution

### Time series analysis

1. Siries of datapoints over time
2. discrete samples taken over dicrete point of time
3. Seasonality - its a feature in your data that shows changes as per seasons
4. Trends - its a feature in your data that shows patterns as per time
5. Noise - changes/variations that are random in nature
6. **seasonality + trends + noise = TimeSeries**  --> use additive models because in this seasonal variation is constant
7. seasonality * trends * noise - trends = amplifies seasonlity and noise  --> use multiplicative models because seasonal variation increases as per trends

### AWS Athena

1. A querying service over s3
2. Data is not loaded from s3, it stays in s3
3. serverless
4. Works on structured, unstructured, semistructured data
5. Athena uses Presto technology underneath
6. Supportin data fromats - csv, orc, parquet, json, avro
7. Ex applications - adhoc querying of weblogs, analyzing cloudtrail/cloudwatch logs in s3
8. Integration with jupyter, quicksight
9. $5 per TB scanned.
10. Cost can be reduced further if we use columnar data (orc, parquet) and also compressing data is useful
11. Transport Layer Security (TLS) between data transfer to/from athena and s3
12. Athena is NOT for visualization (use quicksight), also NOT for ETL (use glue ETL)

### AWS Quicksight

1. Fast and easy cloud-powered business analytics service
2. For visualization and adhoc analysis on all levels, not totally developer oriented
3. serverless
4. pay as you use
5. quickly get business insights from adhoc data on browser/mobile anywhere
6. Data sources - s3 (csv, etc), redshift, dynamodb, rds, athena, or any data sitting on ec2, or any log sitting in ec2 or s3
7. Data preparation aloows simple ETL not all ETL like glue
8. Quicksight is built over technology named **Spice**
9. spice is a superfast, parallel, in-memory calculation engine
10. It uses in-memory columnar storage, accelerates interactive queries on large dataset
11. highly available, scalable, durable, 10gb spice per user
12. Machine Learning feature named **Insights**
13. Machine Learning Insights - anaomaly detection, forecasting, autonarratives
14. ML insights uses inbuilt model named random cut forest derived from random forest
15. Quicksight is very costly, hence rarely used
16. Quicksight visual types -, barchart, pie graph, line graph, autograph, treemap, heatmap, scatter plot, pivot tables, stories

## 12/08/2021

### AWS Elastic MapReduce (EMR)

1. EMR is built over hadoop, runs on ec2
2. Preinstalled techs come with this like - spark, flink, presto, hive, hbase
3. Also includes EMR notebooks running on EMR clusters
4. Used when we have to map and reduce massive dataset before training them
5. EMR provides methods to distribute loads while processing the data paralelly
6. An EMR cluster consists of nodes -
   1. **Master node** - compulsary node, manages the cluster, single ec2 instance, monitors health of the cluster, instances used are - m4.xlarge if nodes > 50, else m4.large if nodes < 50
   2. **Code node** - compulsary node when the cluster is multi layered, runs tasks, hosts hdfs data, can be scaled up/down but with risk, instances used are - m4.large is good for performance, t2.medium for external dependency apps like webcrawler, cluster computing instances if NLP work
   3. **Task node** - optional node, runs task, does not host data, no risk of data while scaling, only used for computational purpose, good for spot instances. (instances can be removed/added), instance same as code node plus spot instances
7. 2 types - **transient cluster**(turn off cluster after task completion, used when task set is predefined) vs **long running cluster**(used when we dont know what to do and expect, also used in interatcive applications, terminate cluster manually after done.)
8. Integration with ec2, cloudwatch, vpc, s3, cloudtrail, IAM, Data pipeline
9. **EMR storage** - hdfs, emrfs (s3 or dynamo), local file system, ebs (elastic block store)
10. EMR charges by the hour, plus ec2 charges
11. **Hadoop** consistes of -
    1. HDFS - Distributed File system, stores multpile copies of data in distributed clusters or node so that data is not lost
    2. YARN - Yet another resource negotiator, centrally manages cluster
    3. MapReduce - software framework to easily write vast amount of data paralally, icludes mapper function(used for transforming; mapping and preparing data) and reduce function(used for compacting and aggregating data into one whole dataset)
    4. Spark - These days Apache spark has largely taken the place of mapreduce, faster alternative to mapreduce, in-memory caching, more smart than mapreduce for data processing, spped, code reuse for batch processing and ml processing
    5. Spark components - Spark streaming, Spark sql, MLlib, GraphX
    6. **Spark MLlib** introduces ml libraries, data is stored distributedly and scalable, parallel processing.
    7. Models include - classification (NaiveBayes, LogisticRegression), Regression, Decision Tress, Clustering (Kmeans), Recommendation Engine (ALS), Topic Modellling (LDA), Ml workflow utilities (pipeline, feature transformation), SVD,PCS, statistics
    8. Spark streaming can be integrated with Kinesis
    9. Spark + Zepplin allows you to run code in notebook env, with all data science tools needed
    10. Spark Notebook, similar concept like above but more integrated with AWS, notebooks are backedup in s3, vps, accessed only via AWS ocnsole, integrated with anaconda, pyspark, python, sparksql, scala
    11. These are all hosted outside EMR, no additional charges for EMR users
    12. EMR secuirty - IAM policies, IAM roles, Kerberos, SSH

## 17/08/2021

### Feature Engineering

1. Feature engineering refers to trimming down our features or create new features or transform some features as required by our problem statement
2. Features are nothing but the attributes of our data to be trained. For ex - if the problem is to detect anomalies in the web requests, then the features would be timestamp, url, ip_address, countrycode, user_agent, etc
3. We cannot just throw raw data to model and expect good results; we need to transform our features using scaling, normalization, handling missing data, handle dimensionality, etc
4. Andrew Ng- "Applied machine learning is basically feature engineering"
5. **Why we do Feature engineering - Because we have curse of dimensionality**
   1. Each feature is an axis, which means 1 dimension.
   2. Now if we say, y is an output(salary) and x is an input(age)., it means the dimesion is 2
   3. But when we add more features like yrs of experience, designation, location, etc, axis increases and hence the dimension increases.
   4. More the dimensions, sparse the relevant information to predict something, more the space to look for proper solution. This results in bad output.
   5. **To reduce dimesion, we have PCA and K-Means**
6. Ways of Feature Engineering -
   1. **Imputation of Missing Data**
      1. **Mean Replacement** - df.fillna(df.mean())
         1. Fill mean values if data is missing in numerical features
         2. Median is better than mean, if we have outliers in our data. For ex, some people won't specify their salary in person data, then it is outliers, don't use mean use median.
         3. Fast and easy
         4. But, works only on column data, hence misses corelation with row data. Also does not work on categorical data
         5. Not very accurate
      2. **Dropping** - df.dropna()
         1. Drops data(row) which has a missing value for any of the coulmn/feature
         2. Not a viable soultion as we might loose imp data
         3. Hence this technique is used only when we have short time, when dropping this data doesn't bias our data
      3. **Machine Learning**
         1. K-means - find most similar(k-nearest) rows and average their values, good for numerical data, not good for categorical data
         2. Deep learning - build ml model to impute data, good for numerical plus categorical data, but high in computation
         3. Regression - find linear/nonlinear relationship between missing features and other features, MICE - Multiple Imputation by Chained Equations
   2. **Question** Which is the best way to impute missing data?
      Best is to get more complete data, If not then Machine learning techniques like - best is MICE, then Deep Learning, then K-means
   3. **Handling Unbalanced Data**
      1. Unbalanced means large difference between positive and negative data.
      2. For ex- fraud detection will have less data for frauds and more data for normal transactions.
      3. Also, positive does not mean normal transactions, it means the thing you are trying to detect has happend. If you are detecting fraud, then fraud is positive, if you are detecting normal transaction, then that is positive. This is very inportant in understanding confusion matrix.
      4. **Oversampling** -
         1. duplicate values randomly in lesser class data
      5. **Undersampling** -
         1. not the best practice, as we might lose important information
      6. **SMOTE** -
         1. Synthetic Minority Oversampling Technique
         2. Artificially create new samples using K-nn
         3. Knn will find the nearest neighbour and just get mean of the result
         4. It generates new data for undersampled class and reduce data fro oversampled class
      7. **Adjust Threshold**
         1. Mostly works in cases where we get prediction score, this score helps to find a particular threshold beyond which we can say fraud or not fraud.
         2. For ex - if we want to reduce FP, increase threshold. It will guarantee reduced FPs but might increase FNs.
