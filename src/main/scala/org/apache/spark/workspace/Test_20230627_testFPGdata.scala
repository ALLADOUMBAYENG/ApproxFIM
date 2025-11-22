package org.apache.spark.workspace

import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.RspContext.SparkSessionFunc
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import smile.association.{ItemSet, fpgrowth}

import java.util.stream.{Collectors, Stream}
import scala.collection.mutable
import scala.collection.JavaConverters._

object Test_20230627_testFPGdata {


  def main(args: Array[String]): Unit = {

    // TODO 构建spark环境
    val sparkconf = new SparkConf().setAppName("Test_Smile")
      .setMaster("local[*]")
      .set("spark.executor.memory","16g")
      .set("spark.executor.cores","1")
      .set("spark.driver.memory","8g")
      .set("spark.driver.cores","1")
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()
    println("------------环境配置成功---------------")

    // 开始时间
    val startTime = System.nanoTime()

//    val transactions = spark.rspRead.parquet("datas/1TdataDemo.parquet")

    val value10 = spark.rspRead.text("datas/1txtFPGdata").rdd

    val value20: RDD[String] = value10.map((f: Row) => f.mkString(" "))


    val transaction: RDD[Array[Int]] = value20.map((_: String).split(" ").map(_.toInt))

    println(transaction.count())

    val l:Long  = transaction.count()

    println(transaction.getNumPartitions)

    //    获取每个分区的训练计时和stream结果
    val value: RDD[Stream[ItemSet]] = transaction.mapPartitions(arr => {
      val array: Array[Array[Int]] = arr.toArray
      val partitionRes: Stream[ItemSet] = fpgrowth((l * 0.15).toInt, array)
      Iterator.single(partitionRes)
    })
    val value2: RDD[ItemSet] = value.mapPartitions((stream: Iterator[Stream[ItemSet]]) => {
      //迭代器里只有一个stream.Stream[ItemSet]
      val elem: Stream[ItemSet] = stream.next()
      val buf: mutable.Buffer[ItemSet] = elem.collect(Collectors.toList[ItemSet]).asScala
      buf.iterator
    })
    //将Itemset对象转成（频繁项集，出现次数）的KV对
    val value1: RDD[(String, Int)] = value2
      .filter(item => item.items.length > 1)
      .map((item: ItemSet) => (item.items.toList.sorted.mkString("{", ",", "}"), item.support))
      .cache()

    //
    val map: Map[String, Int] = value1.map(item => (item._1, 1)) //（频繁项集，1）
      .reduceByKey(_ + _) //（频繁项集，出现该频繁项集的分区数）
      .filter(f => f._2 >= 1) //（投票）
      .collect() //collect放到集合里
      .toMap

    val value3: RDD[(String, Int)] = value1
      .filter(f => map.contains(f._1))
      .combineByKey(v => v, (t: Int, v: Int) => t + v, (t: Int, v: Int) => t + v)
      .map(item => {
        (item._1, (item._2 / map(item._1)))
      })

//    value3.map(x => x._1 + ": " + x._2).repartition(1).saveAsTextFile(path)

    value3.foreach(println)

    // 结束时间
    val endTime = System.nanoTime()
    // 计算执行时间（以毫秒为单位）
    val executionTimeMs = (endTime - startTime) / 1000000

    // 打印执行时间
    println(s"程序执行时间：$executionTimeMs 毫秒")

    spark.close()

  }


}
