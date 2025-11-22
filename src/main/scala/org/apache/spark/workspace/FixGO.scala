package org.apache.spark.workspace

import org.apache.spark.SparkConf
import org.apache.spark.mlutils.GO_Strategy.mergeFPGrowth
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.RspContext._
import org.apache.spark.sql.{Row, RspDataset, SparkSession}
import smile.association.{ItemSet, fpgrowth}
import smile.classification.{DecisionTree, RandomForest, cart, randomForest}
import smile.data.DataFrame
import smile.data.formula.Formula

import java.util.stream.{Collectors, Stream}
import org.apache.spark.mlutils.dataWrapper._
import org.apache.spark.rdd.RDD

import scala.collection.JavaConverters.asScalaBufferConverter
import scala.collection.mutable


object FixGO {
  def main(args: Array[String]): Unit = {

    // TODO 构建spark环境
    val sparkconf = new SparkConf().setAppName("Test_Smile").setMaster("local[*]")
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()
    println("------------环境配置成功---------------")

    val transactions: RspRDD[Array[Array[Int]]] = spark.rspRead.parquet("datas/transaction_dataset_30.parquet")
      .dataWrapper(Smile_Parquet_FPG)

    val leng: Int = transactions.partitions.length

    println("------数据加载成功-------")

    val localTable: RspRDD[Stream[ItemSet]] = transactions.LO(trainDF =>
      fpgrowth((trainDF.length * 0.15).toInt, trainDF)
    )

    val value: RspRDD[(String, Int)] = localTable.GO(mergeFPGrowth)

    value.foreach(println)

//    val itemSetRDD: RDD[ItemSet] = localTable.mapPartitions((stream: Iterator[Stream[ItemSet]]) => {
//      //迭代器里只有一个stream.Stream[ItemSet]
//      val elem: Stream[ItemSet] = stream.next()
//      val buf: mutable.Buffer[ItemSet] = elem.collect(Collectors.toList[ItemSet]).asScala
//      buf.iterator
//    })
//
//    val itemSetWithFreq: RDD[(String, Int)] = itemSetRDD
//      .filter(item => item.items.length > 1)
//      .map((item: ItemSet) => (item.items.toList.sorted.mkString("{", ",", "}"), item.support))
//
//    val itemSetWithFreqAndCount: RDD[(String, (Int,Int))] = itemSetWithFreq.map(elem => {
//      (elem._1, (elem._2, 1))
//    }).reduceByKey(
//      (x, y) => (x._1 + y._1, x._2 + y._2)
//    )
//
//    val value: RDD[(String, (Int, Int))] = itemSetWithFreqAndCount.filter(
//      elem => {
//        elem._2._2 >= leng * 0.5
//      }
//    )
//    value.foreach(println)
//
//    println(localTable.partitions.length)




    spark.close()
  }
}
