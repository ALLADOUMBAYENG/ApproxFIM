import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import scala.math._

object temp {

  // 初始化消息
  def initializeMessages(similarities: RDD[((Int, Int), Double)]): RDD[((Int, Int), (Double, Double))] = {
    similarities.map { case (pair, similarity) =>
      (pair, (0.0, similarity))
    }
  }

  def updateResponsibilities(
                              messages: RDD[((Int, Int), (Double, Double))],
                              dampingFactor: Double
                            ): RDD[((Int, Int), Double)] = {
    messages
      .groupBy { case ((i, _), _) => i }
      .flatMap {
        case (i, groupedMessages) =>
          val maxSimilarity = if (groupedMessages.nonEmpty) {
            groupedMessages.map(_._2._2).max
          } else {
            0.0 // 你可以根据具体需求设置合适的默认值
          }
          groupedMessages.map { case ((i, k), (r, a)) =>
            val otherMax = if (groupedMessages.filter(_._1._2 != k).nonEmpty) {
              groupedMessages.filter(_._1._2 != k).map(_._2._2).max
            } else {
              0.0 // 你可以根据具体需求设置合适的默认值
            }
            ((i, k), dampingFactor * r + (1 - dampingFactor) * (a + otherMax - maxSimilarity))
          }
      }
  }

  // 更新可用度
  def updateAvailabilities(
                            responsibilities: RDD[((Int, Int), Double)],
                            dampingFactor: Double
                          ): RDD[((Int, Int), Double)] = {
    responsibilities
      .groupBy { case ((i, _), _) => i }
      .flatMap {
        case (i, groupedResponsibilities) =>
          groupedResponsibilities.map {
            case ((i, k), r) =>
              val sum = groupedResponsibilities.filter(_._1._2 != k).map(_._2).filter(_ > 0).sum
              ((i, k), dampingFactor * sum + (1 - dampingFactor) * (if (i == k) sum else min(0.0, sum + r)))
          }
      }
  }

  // 计算聚类中心
  def computeClusterCenters(
                             availabilities: RDD[((Int, Int), Double)],
                             responsibilities: RDD[((Int, Int), Double)]
                           ): RDD[(Int, Int)] = {
    availabilities.join(responsibilities)
      .map { case ((i, k), (a, r)) => (i, (k, a + r)) }
      .groupByKey()
      .mapValues(_.maxBy(_._2)._1)
  }

  def runMessagePassingClustering(
                                   similarities: RDD[((Int, Int), Double)],
                                   maxIterations: Int = 100,
                                   dampingFactor: Double = 0.5
                                 ): RDD[(Int, Int)] = {
    var messages = initializeMessages(similarities)

    for (_ <- 0 until maxIterations) {
      val responsibilities = updateResponsibilities(messages, dampingFactor)
      val availabilities = updateAvailabilities(responsibilities, dampingFactor)
      messages = availabilities.join(responsibilities).mapValues { case (a, r) => (r, a) }
    }

    val availabilities = messages.mapValues(_._2)
    val responsibilities = messages.mapValues(_._1)

    computeClusterCenters(availabilities, responsibilities)
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("MessagePassingClustering").setMaster("local")
    val sc = new SparkContext(conf)

    // 示例数据：计算两两点之间的相似度 (假设sim是相似度)
    val similarities: RDD[((Int, Int), Double)] = sc.parallelize(Seq(
      ((1, 2), 0.9),
      ((1, 3), 0.4),
      ((1, 4), 0.5),
      ((2, 3), 0.3),
      ((2, 4), 0.2),
      ((3, 4), 0.8)
    ))

    // 运行Message Passing Clustering
    val clusters = runMessagePassingClustering(similarities)

    // 输出聚类结果
    clusters.collect().foreach { case (point, cluster) =>
      println(s"Point $point is in cluster $cluster")
    }

    sc.stop()
  }
}
