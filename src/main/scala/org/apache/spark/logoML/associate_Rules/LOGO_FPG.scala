package org.apache.spark.logoML.associate_Rules

import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.RspContext.RspRDDFunc
import smile.association.{ItemSet, fpgrowth}

import java.util.stream.Stream


class LOGO_FPG(rdd: RspRDD[Array[Array[Int]]]) {

  val transactions: RspRDD[Array[Array[Int]]] = rdd

  def logoFPGrowth(minSupport: Int, itemsets: Array[Array[Int]]): Stream[ItemSet] = {
    fpgrowth(minSupport, itemsets)
  }

  def trainer(minSupport: Double): RspRDD[Stream[ItemSet]] = {
    transactions.LO(trans =>
      fpgrowth((trans.length * minSupport).toInt, trans)
    )
  }
}

object LOGO_FPG {
  def logoFPGrowth(minSupport: Int, itemsets: Array[Array[Int]]): Stream[ItemSet] = {
    fpgrowth(minSupport, itemsets)
  }
}