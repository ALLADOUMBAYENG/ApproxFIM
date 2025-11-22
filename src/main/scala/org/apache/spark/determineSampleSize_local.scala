package org.apache.spark

import org.apache.spark.ml.fpm.FPGrowth
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.{RspRDD, SonPartitionCoalescer}
import org.apache.spark.sql.RspContext._
import org.apache.spark.sql.functions.col
import org.apache.spark.sql._

import java.util.Date

object determineSampleSize_local {
  def main(args: Array[String]): Unit = {

    // TODO 构建spark环境
    val sparkconf = new SparkConf().setAppName("Test_Smile").setMaster("local[*]")
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()
    println("------------Environment configured successfully---------------")
    val now_start: Date=new Date()
    import spark.implicits._
    var i = 0
    val MST = 0.02
    var rmax: Double = 0.0
    var rmaxWS: Double = 0.0
    val numberOfblocks = 12
    val error: Double = 0.05
    val delta: Double = 0.01

    // 定义一个DataFrame集合，用于存储每个文件挖掘出的频繁项集
    var allFrequentItemsets: DataFrame = spark.emptyDataFrame
    var allFrequentItemsetsWS: DataFrame = spark.emptyDataFrame
    ///user/caimeng/Items_10E_10000_100000.parquet
    //    /user/caiyongda/realWorldDataset/TH2
    val value: DataFrame = spark.read.parquet("data/TH2_260W_260_26W")
    val RSPdata: RspRDD[Row] = spark.rspRead.parquet("data/TH2_260W_260_26W").rdd
    val array: Array[Int] = scala.util.Random.shuffle((0 until RSPdata.getNumPartitions).toList).toArray
    while(i<numberOfblocks) {
      val value1: RDD[Row] = RSPdata.coalesce(1, false, Option(new SonPartitionCoalescer(Array(array(i)))))
      //.repartition(5).coalesce(1, false, Option(new SonPartitionCoalescer(Array(0))))
      val dataset: DataFrame = spark.createDataFrame(value1, value.schema)
      print("数据量"+dataset.count())
      val fpgrowth = new FPGrowth().setItemsCol("items").setMinSupport(MST).setMinConfidence(0.6)
      val model = fpgrowth.fit(dataset)
      //calculate rmax and Is with single frequent itemset
      val FIM: DataFrame = model.freqItemsets
      if (allFrequentItemsets.isEmpty) {
        allFrequentItemsets = FIM.select("items")
      } else {
        allFrequentItemsets = allFrequentItemsets.union(FIM.select("items")).select("items").distinct()
      }
      val FIMsup: DataFrame = FIM.withColumn("supportValue", col("freq") / dataset.count())
      val stats: DataFrame = FIMsup.select("supportValue").describe()
      val maxValue = (stats.filter($"summary" === "max").select("supportValue").collect()(0)(0)).toString.toDouble
      if (rmax<maxValue) {
        rmax = maxValue
      }
      print("\n")
      print("第"+i.toString+"次rmax结果："+rmax)
      print("\n")
      print("第"+i.toString+"次FIMsup结果："+FIMsup.count())
      print("\n")

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
      print("第"+i.toString+"次FIMWSsup结果："+FIMWSsup.count())
      print("\n")

      i = i+1
    }
    val distinctItemsets: Dataset[Row] = allFrequentItemsets
      .select("items")
      .distinct()
    val S: Double = rmax*rmax/(2*error*error)*math.log(2*distinctItemsets.count()/delta)
    val distinctItemsetsWS: Dataset[Row] = allFrequentItemsetsWS
      .select("items")
      .distinct()
    val SWS: Double = rmaxWS*rmaxWS/(2*error*error)*math.log(2*distinctItemsetsWS.count()/delta)
    val now_end: Date=new Date()
    val runningTime = now_end.getTime - now_start.getTime
    print("\n"+"================================================================="
      +"\n"+"parameters setting:" + "MST="+ MST +";numberOfDataBlocks="+numberOfblocks+";error="+error+";delta="+delta
      +"\n"+"rmax (include single itemset):"+rmax
      +"\n"+"IS result (include single itemset):"+distinctItemsets.count().toString
      +"\n"+"Sample size (include single itemset)"+S
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
