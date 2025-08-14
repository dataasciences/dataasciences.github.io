---
title: "The Birth of Spark"
excerpt: "How MapReduce Gave Way to Spark"
categories:
  - Big Data 
tags:
  - Software Development
  - Big Data
  - Spark
sidebar:
  - nav: docs      
classes: wide
show_ads: true
---

In order to understand Spark, it helps to understand its History. If you have worked on Spark, you would have come across the term MapReduce. Before Spark, there was MapReduce. 

MapReduce is a resilient distributed processing framework which enabled Google to index the exploding volume of content on the web, across large clusters of commodity servers. 

![](https://github.com/dataasciences/dataasciences.github.io/blob/master/assets/images/map_reduce_1.png?raw=true)

In this Section, we will look at the core concepts that underlie MapReduce and how Spark came into existence. There are 3 core concepts to how MapReduce works;

## MapReduce Core Concepts

1) Distribute data
  
* When a data file is uploaded into the cluster, it is split into chunks, called data blocks, and distributed amongst the data nodes and replicated across the cluster.

2) Distribute computation
  
* Users specify a map function that processes a key/value pair to generate a set of intermediate key/value pairs and a reduce function that merges all intermediate values associated with the same intermediate key. Programs written in this functional style are automatically parallelized and executed on a large cluster of commodity machines in the following way: 

  * The mapping process runs on each assigned data node, working only on its block of data from a distributed file. 
  * The results from the mapping processes are sent to the reducers in a process called "shuffle and sort": key/value pairs from the mappers are sorted by key, partitioned by the number of reducers, and then sent across the network and written to key sorted "sequence files" on the reducer nodes. The reducer process executes on its assigned node and works only on its subset of the data (its sequence file). The output from the reducer process is written to an output file.

3) Tolerate faults

  * Both data and computation can tolerate failures by failing over to another node for data or processing.

## MapReduce WordCount Example

![](https://github.com/dataasciences/dataasciences.github.io/blob/master/assets/images/map_reduce_2.png?raw=true)

Some iterative algorithms, like PageRank, which Google used to rank websites in their search engine results, require chaining multiple MapReduce jobs together, which causes a lot of reading and writing to disk. When multiple MapReduce jobs are chained together, for each MapReduce job, data is read from a distributed file block into a mapping process, written to and read from a SequenceFile in between, and then written to an output file from a reducer process.

![](https://github.com/dataasciences/dataasciences.github.io/blob/master/assets/images/map_reduce_3.png?raw=true)

## Birth of Spark

After MapReduce was published, Spark came into existence as a project within the Apache Foundation. The goal was to keep the benefits of MapReduce's scalable, distributed, 
fault-tolerant processing framework while making it more efficient and easier to use. The advantages of Spark over MapReduce are;

* Spark executes much faster by caching data in memory across multiple parallel operations, whereas MapReduce involves more reading and writing from disk.
* Spark runs multi-threaded tasks inside of JVM processes, whereas MapReduce runs as heavier-weight JVM processes. This gives Spark faster startup, better parallelism, 
  and better CPU utilization.
* Spark provides a richer functional programming model than MapReduce.
* Spark is especially useful for parallel processing of distributed data with iterative algorithms.

The following diagram shows how a Spark Application runs on a Cluster,

![](https://github.com/dataasciences/dataasciences.github.io/blob/master/assets/images/spark_arch.png?raw=true)

* A Spark application runs as independent processes, coordinated by the SparkSession object in the driver program.
* The resource or cluster manager assigns tasks to workers, one task per partition.
* A task applies its unit of work to the dataset in its partition and outputs a new partition dataset. Because iterative algorithms apply operations repeatedly to data, they benefit from caching datasets across iterations.
* Results are sent back to the driver application or can be saved to disk.

Spark supports the following resource/cluster managers:

**Spark Standalone** – a simple cluster manager included with Spark

**Apache Mesos** – a general cluster manager that can also run Hadoop applications

**Apache Hadoop YARN** – the resource manager in Hadoop 2

**Kubernetes** – an open-source system for automating deployment, scaling, and management of containerized applications 

Spark also has a local mode, where the driver and executors run as threads on your computer instead of a cluster, which is useful for developing your applications from a 
personal computer.

## What Does Spark Do?

Spark is capable of handling several petabytes of data at a time, distributed across a cluster of thousands of cooperating physical or virtual servers. It has an extensive set of developer libraries and APIs and supports languages such as Java, Python, R, and Scala; its flexibility makes it well-suited for a range of use cases. Spark is often used with distributed data stores such as MapR-XD, Hadoop’s HDFS, and Amazon’s S3, with popular NoSQL databases such as MapR-DB, Apache HBase, Apache Cassandra, and MongoDB, and with distributed messaging stores such as MapR-ES and Apache Kafka.

## Use cases of Spark

*Stream processing:*

From log files to sensor data, application developers are increasingly having to cope with “streams” of data. This data arrives in a steady stream, often from multiple sources simultaneously. While it is certainly feasible to store these data streams on disk and analyze them retrospectively, it can sometimes be sensible or important to process and act upon the data as it arrives. Streams of data related to financial transactions, for example, can be processed in real-time to identify – and refuse – potentially fraudulent transactions.

*Machine learning:*

As data volumes grow, Machine learning approaches become more feasible and increasingly accurate. Software can be trained to identify and act upon 
triggers within well-understood data sets before applying the same solutions to new and unknown data. Spark’s ability to store data in memory and rapidly run repeated queries makes it a good choice for training machine learning algorithms. Running broadly similar queries again and again, at scale, significantly reduces the time required to go through a set of possible solutions in order to find the most efficient algorithms.
 
*Interactive analytics:*

Rather than running pre-defined queries to create static dashboards of sales or production line productivity or stock prices, business analysts and data 
scientists want to explore their data by asking a question, viewing the result, and then either altering the initial question slightly or drilling deeper into results. This interactive query process requires systems such as Spark that are able to respond and adapt quickly.

*Data integration:*

Data produced by different systems across a business is rarely clean or consistent enough to simply and easily be combined for reporting or analysis. 
Extract, transform, and load (ETL) processes are often used to pull data from different systems, clean and standardize it, and then load it into a separate system for analysis. Spark (and Hadoop) are increasingly being used to reduce the cost and time required for this ETL process. 

## Reasons to Choose Spark

*Simplicity*: 

Spark’s capabilities are accessible via a set of rich APIs, all designed specifically for interacting quickly and easily with data at scale. These APIs are well-
documented and structured in a way that makes it straightforward for data scientists and application developers to quickly put Spark to work

*Speed:* 

Spark is designed for speed, operating both in memory and on disk. Using Spark, a team from Databricks tied for first place with a team from the University of California, 
San Diego, in the [2014 Daytona GraySort benchmarking challenge] (https://spark.apache.org/news/spark-wins-daytona-gray-sort-100tb-benchmark.html). The challenge involves 
processing a static data set; the Databricks team was able to process 100 terabytes of data stored on solid-state drives in just 23 minutes, and the previous winner took 72 minutes by using Hadoop and a different cluster configuration. Spark can perform even better when supporting interactive queries of data stored in memory. In those situations, there are claims that Spark can be 100 times faster than Hadoop’s MapReduce.

*Support:* 

Spark supports a range of programming languages, including Java, Python, R, and Scala. Spark includes support for tight integration with a number of leading storage 
solutions in the Hadoop ecosystem and beyond, including MapR (file system, database, and event store), Apache Hadoop (HDFS), Apache HBase, and Apache Cassandra. 
Furthermore, the Apache Spark community is large, active, and international. A growing set of commercial providers, including Databricks, IBM, and all of the main Hadoop 
vendors, deliver comprehensive support for Spark-based solutions. 

Although, as of 2024, Spark is widely used, knowing its background and architecture, as we saw above will help in understanding the intricate components involved and in improving the performance of your Big data Application. 


