package org.apache.spark

import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.RspContext.{RspDatasetFunc, RspRDDFunc, SparkSessionFunc}
import org.apache.spark.sql.{Dataset, Row, RspDataset, SparkSession}
import smile.association.{ItemSet, fpgrowth}
import smile.data.DataFrame

import java.util.stream.{Collectors, Stream}
import scala.collection.JavaConverters.asScalaBufferConverter
import scala.collection.mutable
import scala.util.Random

object DemoForFPG_P {
  def main(args: Array[String]): Unit = {

    // TODO 构建spark环境
    val sparkconf = new SparkConf().setAppName("Test_Smile").setMaster("local[*]")
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()
    println("------------环境配置成功---------------")

//    // TODO 数据加载并进行数据封装
//    val loadedData = spark.rspRead.parquet("datas/transaction_dataset_30.parquet")
//    println("------数据加载成功-------")
//
//    val inputData: RspDataset[Row] = loadedData.getSubDataset(5)
//
//    println(loadedData.getSubDataset(1).count())
//    println(loadedData.getSubDataset(2).count())
//    println(loadedData.getSubDataset(3).count())
//    println(loadedData.getSubDataset(4).count())
//    println(loadedData.getSubDataset(5).count())
//
//    var jobs: Array[(Int, Array[Int])] = null
//    val partitionsList = List.range(0, inputData.rdd.getNumPartitions)
//
//    jobs = Array((1,Array(1,2,3,4)))
//
//    val value:RDD[String] = inputData.rdd.map((f: Row) => f.mkString(" "))
//
//    def etl(data: RDD[String]): RDD[Array[Array[Int]]] = {
//      //把数据通过空格分割
//      data.map(_.split(" ").map(_.toInt)).glom()
//    }
//
//    val transaction: RDD[Array[Int]] = inputData.rdd.map(_.get(0).asInstanceOf[mutable.WrappedArray[Int]].toArray)
//
//    val value1: RDD[Array[Array[Int]]] = transaction.glom()
//
//    val value2: RspRDD[Array[Array[Int]]] = new RspRDD(value1)
//
//    val value3: RspRDD[(Stream[ItemSet], Int)] = value2.LO(
//      trainDF =>
//        (fpgrowth(2200, trainDF), trainDF.length)
//    )
//
////    //
//    val value4 = value3.mapPartitions((stream: Iterator[(Stream[ItemSet], Int)]) => {
//      //迭代器里只有一个stream.Stream[ItemSet]
//      val elem: (Stream[ItemSet], Int) = stream.next()
//      val buf: mutable.Buffer[ItemSet] = elem._1.collect(Collectors.toList[ItemSet]).asScala
//      buf.iterator
//    })
//
//    val value8 = value3.map(
//      x => {
//        (x._1.collect(Collectors.toList[ItemSet]).asScala.iterator, x._2)
//      }
//    )
//
//    value8.map(
//      x=>{
//        x._1.map(item => {
//          (item.items.sorted.mkString("{", ",", "}"),item.support)
//        }
//        )
//        (x._1,x._2)
//      }
//    )
//
////    value8.filter(item => item._1.length > 1).
////      map(item =>
////        (item._1.toList.sorted.mkString("{", ",", "}"), item.support)
////      )
//
//
//    //将Itemset对象转成（频繁项集，出现次数）的KV对
////    val value5: RDD[(String, Int)] = value4
////      .filter(item => item.items.length > 1)
////      .map((item: ItemSet) => (item.items.toList.sorted.mkString("{", ",", "}"), item.support))
////
////    val value6:RDD[(String, (Int, Int))] = value5.map(x => {
////      (x._1, (x._2, 1))
////    })
////
////    val value7: RDD[(String, (Int, Int))] = value6.reduceByKey(
////      (x, y) => (x._1 + y._1, x._2 + y._2)
////    )
//
////    value7.foreach(println)
//
//
//
//
//
//
//    //    value4.foreach(println)
//
//
//
//
//
//
//    //    def FPG(trainFrame2: Array[Int]) = {
//    //      fpgrowth(2200,trainFrame2.toArray)
//    //      trainFrame2.toArray
//    //    }
//
//    //    transactionRDD.LO(FPG)
//
//    //    val value3: RDD[Stream[ItemSet]] = transaction.mapPartitions(arr => {
//    //      val array: Array[Array[Int]] = arr.toArray
//    //      val partitionRes: Stream[ItemSet] = fpgrowth(2200, array)
//    //      Iterator.single(partitionRes)
//    //    })
//    //    val value2: RDD[ItemSet] = value3.mapPartitions((stream: Iterator[Stream[ItemSet]]) => {
//    //      //迭代器里只有一个stream.Stream[ItemSet]
//    //      val elem: Stream[ItemSet] = stream.next()
//    //      val buf: mutable.Buffer[ItemSet] = elem.collect(Collectors.toList[ItemSet]).asScala
//    //      buf.iterator
//    //    })
//    //    //将Itemset对象转成（频繁项集，出现次数）的KV对
//    //    val value4: RDD[(String, Int)] = value2
//    //      .filter(item => item.items.length > 1)
//    //      .map((item: ItemSet) => (item.items.toList.sorted.mkString("{", ",", "}"), item.support))
//    //
//    //    value4.foreach(println)
//
//
//
//
//
//
//
//    //    partitionsList.foreach(println)
//
//
//
//
//
//    //    val rdd: RspRDD[Row] = inputData.rdd
//
//
//
//
//
//
//
//
//
//

    spark.close()
  }
}