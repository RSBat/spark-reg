package org.apache.spark.ml.reg

import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.linalg.BLAS.dot
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap}
import org.apache.spark.ml.regression.{RegressionModel, Regressor}
import org.apache.spark.ml.util._
import org.apache.spark.ml.PredictorParams
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{Dataset, Encoder}

trait LinearRegressionParams extends PredictorParams {
  val learningRate = new DoubleParam(this, "learningRate",
    "Learning rate for gradient descent")

  def setLearningRate(value: Double) : this.type = set(learningRate, value)

  setDefault(learningRate -> 0.001)

  val iterations = new IntParam(this, "iterations",
    "Number of iterations of gradient descent")

  def setIterations(value: Int) : this.type = set(iterations, value)

  setDefault(iterations -> 1000)
}

class LinearRegression(override val uid: String) extends Regressor[Vector, LinearRegression, LinearRegressionModel] with LinearRegressionParams
  with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("LinearRegression"))

  override def train(dataset: Dataset[_]): LinearRegressionModel = {

    // Used to convert untyped dataframes to datasets with vectors
    implicit val vencoder : Encoder[Vector] = ExpressionEncoder()
    implicit val dencoder : Encoder[Double] = ExpressionEncoder()

    val vectors: Dataset[(Vector, Double)] = dataset.select(
      dataset($(featuresCol)).as[Vector],
      dataset($(labelCol)).as[Double]
    )

    val dim: Int = AttributeGroup.fromStructField(dataset.schema($(featuresCol))).numAttributes.getOrElse(
      vectors.first()._1.size
    )

    var weights = Vectors.zeros(dim)
    var bias = 0.0

    def add_grads(lhs: (breeze.linalg.Vector[Double], Double),
                  rhs: (breeze.linalg.Vector[Double], Double)) = {
      (lhs._1 + rhs._1, lhs._2 + rhs._2)
    }

    val rowCount = vectors.count()
    for (_ <- 1 to $(iterations)) {
      val full_grad = vectors.rdd.mapPartitions((data: Iterator[(Vector, Double)]) => {
        val agg_grad = data.map(x => {
          val features = x._1
          val label = x._2
          val prediction = features.dot(weights) + bias
          val loss = prediction - label
          val grad = loss * features.asBreeze
          val bias_grad = loss
          (grad, bias_grad)
        }).reduce(add_grads)
        Iterator(agg_grad)
      }).reduce(add_grads)

      weights = Vectors.fromBreeze(weights.asBreeze - $(learningRate) / rowCount * full_grad._1)
      bias = bias - $(learningRate) / rowCount * full_grad._2
    }

    copyValues(new LinearRegressionModel(weights, bias)).setParent(this)

    //    val Row(row: Row) =  dataset
    //      .select(Summarizer.metrics("mean", "std").summary(dataset($(inputCol))))
    //      .first()
    //
    //    copyValues(new LinearRegressionModel(row.getAs[Vector](0).toDense, row.getAs[Vector](1).toDense)).setParent(this)
  }

  override def copy(extra: ParamMap): LinearRegression = defaultCopy(extra)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[reg](override val uid: String,
                                          val weights: Vector,
                                          val bias: Double)
    extends RegressionModel[Vector, LinearRegressionModel] with LinearRegressionParams with MLWritable {

  private[reg] def this(weights: Vector, bias: Double) =
    this(Identifiable.randomUID("LinearRegressionModel"), weights, bias)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(weights, bias), extra)

  override def predict(features: Vector): Double = {
    dot(features, weights) + bias
  }

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val vectors = weights.asInstanceOf[Vector] -> bias.asInstanceOf[Double]

      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder : Encoder[Vector] = ExpressionEncoder()
      implicit val encoderD : Encoder[Double] = ExpressionEncoder()

      val (weights, bias) =  vectors.select(vectors("_1").as[Vector], vectors("_2").as[Double]).first()

      val model = new LinearRegressionModel(weights, bias)
      metadata.getAndSetParams(model)
      model
    }
  }
}
