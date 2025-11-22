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

object DemoForFPG {
  def main(args: Array[String]): Unit = {

    // TODO 构建spark环境
    val sparkconf = new SparkConf().setAppName("Test_Smile").setMaster("local[*]")
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()
    println("------------环境配置成功---------------")

    // TODO 数据加载并进行数据封装
    val loadedData = spark.rspRead.parquet("datas/transaction_dataset_30.parquet")
    println("------数据加载成功-------")

    val inputData: RspDataset[Row] = loadedData.getSubDataset(5)

    var jobs: Array[(Int, Array[Int])] = null
    val partitionsList = List.range(0, inputData.rdd.getNumPartitions)

    jobs = Array((1,Array(1,2,3,4)))

//    for ((sizex, partitions) <- jobs) {
//      val size = Math.ceil(sizex * 0.05).toInt
//      var trainName = "train(size=%d)".format(size)
//      var partitionCount = size
//      val trainRdd: RspRDD[Row] = inputData.rdd.getSubPartitions(2)
//      val l: Long = trainRdd.count()
//      val value: RDD[String] = trainRdd.map((f: Row) => f.mkString(" "))
//      println(value.first())
//      println(trainRdd.first())
//      println(trainName)
//      println("partitionCount: " + partitionCount)
////      SmileFPGrowth.runv(value, vote, (l * minsup / size).toInt, "/user/caimeng/fpg/logov/size_" + sizex * 100 + "_" + minsup)
//    }

    val value:RDD[String] = inputData.rdd.map((f: Row) => f.mkString(" "))

//    txt格式的数据的转换方式
//    val transaction: RDD[Array[Int]] = value.map((_: String).split(" ").map(_.toInt))

//    value.map(x => x.get(0).asInstanceof[mutable.WrappedArray[Int]].toArray)
//
//    transaction.count()

//    val partitionCount: Int = value.getNumPartitions

    def etl(data: RDD[String]): RDD[Array[Array[Int]]] = {
      //把数据通过空格分割
      data.map(_.split(" ").map(_.toInt)).glom()
    }

//    def etl(df: DataFrame): RDD[Array[Int]] = {
//      df.rdd.map(_.get(0).asInstanceof[mutable.WrappedArray[Int]].toArray)
//    }
    val transaction: RDD[Array[Int]] = inputData.rdd.map(_.get(0).asInstanceOf[mutable.WrappedArray[Int]].toArray)

    val value1: RDD[Array[Array[Int]]] = transaction.glom()

    val value2: RspRDD[Array[Array[Int]]] = new RspRDD(value1)

    val value3: RspRDD[Stream[ItemSet]] = value2.LO(
      trainDF =>
        fpgrowth(2200, trainDF)
    )

//
    val value4: RDD[ItemSet] = value3.mapPartitions((stream: Iterator[Stream[ItemSet]]) => {
      //迭代器里只有一个stream.Stream[ItemSet]
      val elem: Stream[ItemSet] = stream.next()
      val buf: mutable.Buffer[ItemSet] = elem.collect(Collectors.toList[ItemSet]).asScala
      buf.iterator
    })


    //将Itemset对象转成（频繁项集，出现次数）的KV对
    val value5: RDD[(String, Int)] = value4
      .filter(item => item.items.length > 1)
      .map((item: ItemSet) => (item.items.toList.sorted.mkString("{", ",", "}"), item.support))

    val value6:RDD[(String, (Int, Int))] = value5.map(x => {
      (x._1, (x._2, 1))
    })

    val value7: RDD[(String, (Int, Int))] = value6.reduceByKey(
      (x, y) => (x._1 + y._1, x._2 + y._2)
    )

    value7.foreach(println)






//    value4.foreach(println)






//    def FPG(trainFrame2: Array[Int]) = {
//      fpgrowth(2200,trainFrame2.toArray)
//      trainFrame2.toArray
//    }

//    transactionRDD.LO(FPG)

//    val value3: RDD[Stream[ItemSet]] = transaction.mapPartitions(arr => {
//      val array: Array[Array[Int]] = arr.toArray
//      val partitionRes: Stream[ItemSet] = fpgrowth(2200, array)
//      Iterator.single(partitionRes)
//    })
//    val value2: RDD[ItemSet] = value3.mapPartitions((stream: Iterator[Stream[ItemSet]]) => {
//      //迭代器里只有一个stream.Stream[ItemSet]
//      val elem: Stream[ItemSet] = stream.next()
//      val buf: mutable.Buffer[ItemSet] = elem.collect(Collectors.toList[ItemSet]).asScala
//      buf.iterator
//    })
//    //将Itemset对象转成（频繁项集，出现次数）的KV对
//    val value4: RDD[(String, Int)] = value2
//      .filter(item => item.items.length > 1)
//      .map((item: ItemSet) => (item.items.toList.sorted.mkString("{", ",", "}"), item.support))
//
//    value4.foreach(println)







//    partitionsList.foreach(println)





//    val rdd: RspRDD[Row] = inputData.rdd











    spark.close()
  }
}