package org.apache.spark

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.fpm.FPGrowth
import org.apache.spark.mlutils.GO_Strategy.mergeFPGrowth
import org.apache.spark.mlutils.dataWrapper.Smile_Parquet_FPG
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.{RspRDD, SonPartitionCoalescer}
import org.apache.spark.sql.RspContext._
import org.apache.spark.sql._
import org.apache.spark.sql.functions.col
import smile.association.{ItemSet, fpgrowth}
import org.apache.spark.RspContext.{NewRDDFunc, RspDatasetFunc, RspRDDFunc}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}

import java.util.Date
import java.util.stream.{Collectors, Stream}
import scala.collection.JavaConverters.asScalaBufferConverter
import scala.collection.mutable
import scala.util.Random

object determineSampleSize_local_test {
  def main(args: Array[String]): Unit = {

    // 获取 Hadoop 文件系统对象
    val hadoopConf = new Configuration()
    val hadoopFs = FileSystem.get(hadoopConf)
    // 指定要检查的目录路径
    val outputPath = new Path("output/data")
    // 检查目录是否存在
    if (hadoopFs.exists(outputPath)) {
      println(s"Directory $outputPath exists. Deleting...")
      // 删除目录
      hadoopFs.delete(outputPath, true)
      println(s"Directory $outputPath has been deleted.")
    } else {
      println(s"Directory $outputPath does not exist.")
    }

    // TODO 构建spark环境
    val sparkconf = new SparkConf().setAppName("DetermineSampleSize").setMaster("yarn")
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()
    println("------------环境配置成功---------------")
    val now_start: Date=new Date()
    import spark.implicits._

    val MST = args(0).toDouble
    val numberOfblocks = args(1).toInt
    val error1 = 0.05
    val error: Double = 0.005
    val delta: Double = 0.05
    val numberofTransactions = 20000

    val dataRead = spark.rspRead.parquet("/user/caimeng/Items_10E_10000_100000.parquet")

//    val value1: RspRDD[Row] = dataRead.rdd.coalesce(numberOfblocks, false, Option(new SonPartitionCoalescer((sampling_Without_Replacement(dataRead.rdd.getNumPartitions, numberOfblocks - 1)).toArray))).toRSP(numberOfblocks * 5)
    val value1: RspRDD[Row] = dataRead.rdd.getRandomPartitions(numberOfblocks).toRSP(numberOfblocks*5)
    spark.createDataFrame(value1, spark.rspRead.parquet("/user/caimeng/Items_10E_10000_100000.parquet").schema).write.parquet("output/data")
    val numberOfData: Double = value1.getSubPartitions(1).count().toDouble

    val transactions: RspRDD[Array[Array[Int]]]  = spark.rspRead.parquet("output/data")
      .dataWrapper(Smile_Parquet_FPG)

    val localTable: RspRDD[Stream[ItemSet]] = transactions.LO(trainDF =>
      fpgrowth((trainDF.length * MST).toInt, trainDF)
    )

    val itemSetRDD: RDD[ItemSet] = localTable.mapPartitions((stream: Iterator[Stream[ItemSet]]) => {
      //迭代器里只有一个stream.Stream[ItemSet]
      val elem: Stream[ItemSet] = stream.next()
      val buf: mutable.Buffer[ItemSet] = elem.collect(Collectors.toList[ItemSet]).asScala
      buf.iterator
    })
    val scheme = StructType(Array(
      StructField("items", StringType, true),
      StructField("freq", IntegerType, true)
    ))

    val itemSetWithFreq: RDD[(String, Int)] = itemSetRDD
      .filter(item => item.items.length > 1)
      .map((item: ItemSet) => (item.items.toList.sorted.mkString("{", ",", "}"), item.support))
    val value: RDD[(String, Int)] = itemSetWithFreq.reduceByKey(math.max(_, _))
    val FIM_temp_Row: RDD[Row] = value.map { case (name, age) => Row(name, age) }
    val FIM_temp_DF: DataFrame = spark.createDataFrame(FIM_temp_Row, scheme)
    val FIM_temp_DF1: DataFrame = FIM_temp_DF.withColumn("supportValue", col("freq") / numberofTransactions)
    val stats: DataFrame = FIM_temp_DF1.select("supportValue").describe()
    val maxValue = (stats.filter($"summary" === "max").select("supportValue").collect()(0)(0)).toString.toDouble
    print(FIM_temp_DF.count())
    print("\n")
    print(maxValue)

    val rmaxWS = (stats.filter($"summary" === "max").select("supportValue").collect()(0)(0)).toString.toDouble


    val SWS: Double = rmaxWS*rmaxWS/(2*error*error)*math.log(2*FIM_temp_DF1.count()/delta)
    val SWS1: Double = rmaxWS*rmaxWS/(2*error1*error1)*math.log(2*FIM_temp_DF1.count()/delta)
    val now_end: Date=new Date()
    val runningTime = now_end.getTime - now_start.getTime
    print("\n"+"================================================================="
      +"\n"+"parameters setting:" + "MST="+ MST +";numberOfDataBlocks="+numberOfblocks+";error="+error+";delta="+delta
      +"\n"+"rmaxWS (exclude single itemset):"+rmaxWS
      +"\n"+"IS result (exclude single itemset)"+FIM_temp_DF1.count()
      +"\n"+"Sample size (exclude single itemset)"+error1 +"\n" +SWS1.toInt
      +"\n"+"Sample size (exclude single itemset)"+error+"\n" +SWS.toInt
      +"\n"+"running time:"+ runningTime +"ms"
      +"\n"+"==================================================================="
      +"\n")


//    val MST = 0.009
//    val numberOfblocks = 20
//    val error1 = 0.05
//    val error: Double = 0.01
//    val delta: Double = 0.05
//    val numberofTransactions = 101910
//
//    val dataRead: RspRDD[Array[Array[Int]]] = spark.rspRead.parquet("data/TH2").dataWrapper(Smile_Parquet_FPG)
//    val datasub: RspRDD[Array[Array[Int]]] = dataRead.getRandomPartitions(numberOfblocks)
//    val localTable: RspRDD[Stream[ItemSet]] = datasub.LO(trainDF =>
//      fpgrowth((trainDF.length * MST).toInt, trainDF)
//    )
//
//    val itemSetRDD: RDD[ItemSet] = localTable.mapPartitions((stream: Iterator[Stream[ItemSet]]) => {
//      //迭代器里只有一个stream.Stream[ItemSet]
//      val elem: Stream[ItemSet] = stream.next()
//      val buf: mutable.Buffer[ItemSet] = elem.collect(Collectors.toList[ItemSet]).asScala
//      buf.iterator
//    })
//    val scheme = StructType(Array(
//      StructField("items", StringType, true),
//      StructField("freq", IntegerType, true)
//    ))
//
//    val itemSetWithFreq: RDD[(String, Int)] = itemSetRDD
//          .filter(item => item.items.length > 1)
//          .map((item: ItemSet) => (item.items.toList.sorted.mkString("{", ",", "}"), item.support))
//    val value: RDD[(String, Int)] = itemSetWithFreq.reduceByKey(math.max(_, _))
//    val FIM_temp_Row: RDD[Row] = value.map { case (name, age) => Row(name, age) }
//    val FIM_temp_DF: DataFrame = spark.createDataFrame(FIM_temp_Row, scheme)
//    val FIM_temp_DF1: DataFrame = FIM_temp_DF.withColumn("supportValue", col("freq") / numberofTransactions)
//    val stats: DataFrame = FIM_temp_DF1.select("supportValue").describe()
//    val maxValue = (stats.filter($"summary" === "max").select("supportValue").collect()(0)(0)).toString.toDouble
//    print(FIM_temp_DF.count())
//    print("\n")
//    print(maxValue)
//
//    val rmaxWS = (stats.filter($"summary" === "max").select("supportValue").collect()(0)(0)).toString.toDouble
//
//
//    val SWS: Double = rmaxWS*rmaxWS/(2*error*error)*math.log(2*FIM_temp_DF1.count()/delta)
//    val SWS1: Double = rmaxWS*rmaxWS/(2*error1*error1)*math.log(2*FIM_temp_DF1.count()/delta)
//    val now_end: Date=new Date()
//    val runningTime = now_end.getTime - now_start.getTime
//    print("\n"+"================================================================="
//      +"\n"+"parameters setting:" + "MST="+ MST +";numberOfDataBlocks="+numberOfblocks+";error="+error+";delta="+delta
//      +"\n"+"rmaxWS (exclude single itemset):"+rmaxWS
//      +"\n"+"IS result (exclude single itemset)"+FIM_temp_DF1.count()
//      +"\n"+"Sample size (exclude single itemset)"+error1 +"\n" +SWS1.toInt
//      +"\n"+"Sample size (exclude single itemset)"+error+"\n" +SWS.toInt
//      +"\n"+"running time:"+ runningTime +"ms"
//      +"\n"+"==================================================================="
//      +"\n")

  }
  def sampling_Without_Replacement(total: Int, subNum: Int) = {
    var arr = 0 to total toArray
    var outList: List[Int] = Nil
    var border = arr.length
    for (i <- 0 to subNum - 1) {
      val index = (new Random).nextInt(border)
      outList = outList ::: List(arr(index))
      arr(index) = arr.last
      arr = arr.dropRight(1)
      border -= 1
    }
    outList
  }

}
