package org.apache.spark.workspace


import org.apache.spark.mlutils.GO_Strategy.mergeFPGrowth
import org.apache.spark.mlutils.dataWrapper.Smile_Parquet_FPG
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.RspContext._
import org.apache.spark.sql.functions.udf
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import smile.association.{ItemSet, fpgrowth}

import java.util.stream.Stream
import scala.collection.mutable


object secondFunc {
  def main(args: Array[String]): Unit = {

    // TODO 构建spark环境
    val sparkConf = new SparkConf().setAppName("Test_Smile").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)
    val spark = SparkSession
      .builder()
      .config(sparkConf)
      .getOrCreate()
    println("------------The environment configuration is successful---------------")

    // TODO 计算全局频率表
    val transactions: RspRDD[Array[Array[Int]]] = spark.rspRead.parquet("datas/transaction_dataset_30.parquet")
      .dataWrapper(Smile_Parquet_FPG)
    val goTable = getGoTable(transactions, 0.15)
     goTable.foreach(println)
    println("-----------------The global table has been obtained-------------------")

    // TODO 获取商品的价格和类别信息
    val goodsTableFrame: DataFrame = spark.read.json("datas/priceT.json")
    val goodsTable: Map[String, (Double, String)] = getGoodsTable(goodsTableFrame)
    // println(goodsTable)
    println("-----------------The product form has been obtained-------------------")

    // TODO 使用SparkSQL做数据分析
    import spark.implicits._
    val frame: DataFrame = goTable.toDF("FIM", "support")
    frame.createOrReplaceTempView("GlobalTable")

    // TODO 构建UDF函数辅助SparkSQL的执行
    def getTotalPrice(name: String): Double = {
      val array1: Array[String] = name.stripPrefix("{").stripSuffix("}").split(",") //{99,100}类型的数据组拆分成String数组
      var totalPrice = 0.0
      for (str <- array1) {
        val tmp: (Double, String) = goodsTable.getOrElse(str, (0, null))
        totalPrice = totalPrice + tmp._1
      }
      BigDecimal(totalPrice).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
    }
    def getIfOrderCategory(name: String): Boolean = {
      val array1: Array[String] = name.stripPrefix("{").stripSuffix("}").split(",") //{99,100}类型的数据组拆分成String数组
      val set: mutable.Set[String] = mutable.Set()
      for (str <- array1) {
        val tmp: (Double, String) = goodsTable.getOrElse(str, (0, null))
        set += tmp._2
      }
      if (set.size == 1) {
        true
      } else {
        false
      }
    }
    def getOrderCategory(name: String) = {
      val array1: Array[String] = name.stripPrefix("{").stripSuffix("}").split(",") //{99,100}类型的数据组拆分成String数组
      val set: mutable.Set[String] = mutable.Set()
      for (str <- array1) {
        val tmp: (Double, String) = goodsTable.getOrElse(str, (0, null))
        set += tmp._2
      }
      set.toArray
    }

    spark.udf.register("TotalPrice", udf(getTotalPrice _))
    spark.udf.register("IfOrderCategory", udf(getIfOrderCategory _))
    spark.udf.register("OrderCategory", udf(getOrderCategory _))

    // TODO 执行SparkSQL获取结果
    val frame1: DataFrame = spark.sql("select FIM,support/10000 as support,TotalPrice(FIM) as TotalPrice,IfOrderCategory(FIM) as IfOrderCategory,OrderCategory(FIM) as OrderCategory from GlobalTable")
    frame1.show(20)
    println(frame1.schema)
    frame1.createOrReplaceTempView("resultTable")


    // TODO 自定义显示各类排序结果
    spark.sql("select FIM,support,TotalPrice,IfOrderCategory,OrderCategory from resultTable Order By support DESC").show(20)
    spark.sql("select FIM,support,TotalPrice,IfOrderCategory,OrderCategory from resultTable Order By TotalPrice DESC").show(20)
    spark.sql("select FIM,support,TotalPrice,IfOrderCategory,OrderCategory from resultTable where IfOrderCategory = true").show(20)

    spark.close()

  }

  // 获取频繁项集
  def getGoTable(trans: RspRDD[Array[Array[Int]]], support: Double): RspRDD[(String, Int)] = {
    val localTable: RspRDD[Stream[ItemSet]] = trans.LO(trainDF =>
      fpgrowth((trainDF.length * support).toInt, trainDF)
    )
    val goTable: RspRDD[(String, Int)] = localTable.GO(mergeFPGrowth)
    goTable
  }

  // 将商品列表从Frame转成Map类型的Table
  def getGoodsTable(goodsTableFrame: DataFrame): Map[String, (Double, String)] = {
    goodsTableFrame.show(3)
    val goodsTableRDD: RDD[Map[String, (Double, String)]] = goodsTableFrame.rdd.map(
      row => {
        Map((row.getAs[String]("name"), (row.getAs[Double]("price"), row.getAs[String]("Type"))))
      }
    )
    val goodsTable: Map[String, (Double, String)] = goodsTableRDD.reduce((map1, map2) => map1 ++ map2)
    goodsTable
  }


}
