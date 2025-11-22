package org.apache.spark.logo

import org.apache.spark.rsp.RspRDD

trait logoML[T,M] {

  /**
   * RspRDD对象
   */

  val content: RspRDD[T]


  /**
   * 标签数据类型
   */

  /**
   *
   * @param rdd : 原始数据RDD
   * @return trainData: RDD[ (Array[Int], Array[ Array[Double] ]) ]
   */

  def dataWrapper() = ???




}
