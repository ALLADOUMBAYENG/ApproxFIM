package org.apache.spark.mlutils

import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.{Row, RspDataset}
import smile.data

import scala.collection.mutable

/**
 * Created by sunxudong on 2023/6/3
 */


object dataWrapper {

  def Smile_Parquet_FPG(inputData: RspDataset[Row]): RspRDD[Array[Array[Int]]] = {

    println("读入parquet文件，向Smile算法库 频繁项集 数据格式兼容....")

    val compatible_Smile_Parquet_FPG: RDD[Array[Array[Int]]] = inputData.rdd
      .map(_.get(0).asInstanceOf[mutable.WrappedArray[Int]].toArray)
      .glom()

    println("--------------")
    new RspRDD(compatible_Smile_Parquet_FPG)
  }

  def Smile_Parquet_Classification(inputData: RspDataset[Row]): RspRDD[smile.data.DataFrame] = {

    println("读入parquet文件，向Smile算法库 分类 数据格式兼容....")

    val compatible_Smile_Parquet_Classification: RDD[data.DataFrame] = inputData.rdd.glom().map(
      f => (
        f.map(r => r.getInt(0)),
        f.map(r => r.get(1).asInstanceOf[DenseVector].toArray)
      )
    ).map(
      f => {
        val labelFrame = data.DataFrame.of(f._1.map(l => Array(l)), "Y")
        val featureFrame = data.DataFrame.of(f._2)
        val trainFrame = featureFrame.merge(labelFrame)
        trainFrame
      }
    )
    new RspRDD(compatible_Smile_Parquet_Classification)
  }


}
