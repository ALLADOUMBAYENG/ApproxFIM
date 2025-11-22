package org.apache.spark

import breeze.linalg.Vector.castFunc
import breeze.linalg.max
import org.apache.spark.ml.fpm.FPGrowth
import org.apache.spark.mlutils.GO_Strategy.mergeFPGrowth
import org.apache.spark.mlutils.dataWrapper._
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.{RspRDD, SonPartitionCoalescer}
import org.apache.spark.sql.RspContext.{RspDatasetFunc, _}
import org.apache.spark.sql.functions.{col, log}
import org.apache.spark.sql.{DataFrame, Dataset, Row, RspDataset, SparkSession, functions}
import smile.association.{ItemSet, fpgrowth}

import java.io.{BufferedWriter, File, FileWriter}
import java.util.Date
import java.util.stream.Stream
import scala.util.Random

object determineSampleSize {
  def main(args: Array[String]): Unit = {


    // TODO 构建spark环境
    val sparkconf = new SparkConf().setAppName("Test_Smile").setMaster("yarn")
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()
    println("------------环境配置成功---------------")
    val now_start: Date=new Date()
    import spark.implicits._
    var i = 0
    val MST = args(0).toDouble//0.005
    var rmax: Double = 0.0
    var rmaxWS: Double = 0.0
    val numberOfblocks = args(1).toInt//5
    val error: Double = args(2).toDouble//0.05
    val delta: Double = args(3).toDouble//0.05

    // 定义一个DataFrame集合，用于存储每个文件挖掘出的频繁项集
    var allFrequentItemsets: DataFrame = spark.emptyDataFrame
    var allFrequentItemsetsWS: DataFrame = spark.emptyDataFrame
    ///user/caimeng/Items_10E_10000_100000.parquet
    //    /user/caiyongda/realWorldDataset/TH2
    val value: DataFrame = spark.read.parquet("data/Kaggle")
    val RSPdata: RspRDD[Row] = spark.rspRead.parquet("data/Kaggle").rdd
    val array: Array[Int] = scala.util.Random.shuffle((0 until RSPdata.getNumPartitions).toList).toArray
    while(i<numberOfblocks) {
      val value1: RDD[Row] = RSPdata.coalesce(1, false, Option(new SonPartitionCoalescer(Array(array(i)))))
//      .repartition(5).coalesce(1, false, Option(new SonPartitionCoalescer(Array(0))))
      val dataset: DataFrame = spark.createDataFrame(value1, value.schema)
      print("数据量"+dataset.count())
      val fpgrowth = new FPGrowth().setItemsCol("items").setMinSupport(MST).setMinConfidence(0.6)
      val model = fpgrowth.fit(dataset)
      //calculate rmax and Is with single frequent itemset
      //      val FIM: DataFrame = model.freqItemsets
      //      if (allFrequentItemsets.isEmpty) {
      //        allFrequentItemsets = FIM.select("items")
      //      } else {
      //        allFrequentItemsets = allFrequentItemsets.union(FIM.select("items")).select("items").distinct()
      //      }
      //      val FIMsup: DataFrame = FIM.withColumn("supportValue", col("freq") / dataset.count())
      //      val stats: DataFrame = FIMsup.select("supportValue").describe()
      //      val maxValue = (stats.filter($"summary" === "max").select("supportValue").collect()(0)(0)).toString.toDouble
      //      if (rmax<maxValue) {
      //        rmax = maxValue
      //      }
      //      print("第"+i.toString+"次rmax结果："+rmax)
      //      print("\n")
      //      print("第"+i.toString+"次FIMsup结果："+FIMsup.count())
      //      print("\n")

      //calculate rmax and Is without single frequent itemset
      val FIMWS: DataFrame = model.freqItemsets.filter(functions.size($"items") > 1)
      if (allFrequentItemsetsWS.isEmpty) {
        allFrequentItemsetsWS = FIMWS.select("items")
      } else {
        allFrequentItemsetsWS = allFrequentItemsetsWS.union(FIMWS.select("items")).select("items").distinct()
      }
      val FIMWSsup: DataFrame = FIMWS.withColumn("supportValue", col("freq") / dataset.count())
      val statsWS: DataFrame = FIMWSsup.select("supportValue").describe()
      val maxValueWS = (statsWS.filter($"summary" === "max").select("supportValue").collect()(0)(0)).toString.toDouble
      if (rmaxWS<maxValueWS) {
        rmaxWS = maxValueWS
      }
      print("第"+i.toString+"次rmaxWS结果："+rmaxWS)
      print("\n")
      //      print("第"+i.toString+"次FIMWSsup结果："+FIMWSsup.count())
      //      print("\n")

      i = i+1
    }
    //    val distinctItemsets: Dataset[Row] = allFrequentItemsets
    //      .select("items")
    //      .distinct()
    //    val S: Double = rmax*rmax/(2*error*error)*math.log(2*distinctItemsets.count()/delta)
    val distinctItemsetsWS: Dataset[Row] = allFrequentItemsetsWS
      .select("items")
      .distinct()
    val SWS: Double = rmaxWS*rmaxWS/(2*error*error)*math.log(2*distinctItemsetsWS.count()/delta)
    val now_end: Date=new Date()
    val runningTime = now_end.getTime - now_start.getTime
    print("\n"+"================================================================="
      +"\n"+"parameters setting:" + "MST="+ MST +";numberOfDataBlocks="+numberOfblocks+";error="+error+";delta="+delta
      //      +"\n"+"rmax (include single itemset):"+rmax
      //      +"\n"+"IS result (include single itemset):"+distinctItemsets.count().toString
      //      +"\n"+"Sample size (include single itemset)"+S
      +"\n"+"rmaxWS (exclude single itemset):"+rmaxWS
      +"\n"+"IS result (exclude single itemset)"+distinctItemsetsWS.count()
      +"\n"+"Sample size (exclude single itemset)"+SWS
      +"\n"+"running time:"+ runningTime +"ms"
      +"\n"+"==================================================================="
      +"\n")

    //    i = 1
    //    while(i<numberOfblocks){
    //      val value: RspDataset[Row] = spark.rspRead.parquet("data/TH1_260W_10_26W")
    //      val nums = value.rdd.getNumPartitions
    //      val array: Array[Int] = scala.util.Random.shuffle((0 until nums).toList).toArray
    //      val value1: RspRDD[Row] = value.rdd.coalesce(1, false, Option(new SonPartitionCoalescer(Array(array(0))))).toRSP(1)
    //      val value2: RspRDD[Array[Array[Int]]] = value.rspRDDtoRspDataset(value1).dataWrapper(Smile_Parquet_FPG)
    //      val localTable: RspRDD[Stream[ItemSet]] = value2.LO(trainDF =>
    //        fpgrowth((trainDF.length * 0.005).toInt, trainDF)
    //      )
    //      print(localTable.count())
    //      val goTable: RspRDD[(String, Int)] = localTable.GO(mergeFPGrowth)
    //      goTable.foreach(println)
    //      print(goTable.count())
    //      i = i+1
    //    }
    //    spark.close()

  }
}
