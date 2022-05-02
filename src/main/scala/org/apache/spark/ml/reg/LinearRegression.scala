package org.apache.spark.ml.reg

import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.param.{BooleanParam, ParamMap}
import org.apache.spark.ml.regression.{RegressionModel, Regressor}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model, PredictorParams}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.{DataType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}

trait LinearRegressionParams extends PredictorParams

class LinearRegression(override val uid: String) extends Regressor[Vector, LinearRegression, LinearRegressionModel] with LinearRegressionParams
  with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("LinearRegression"))

  override def train(dataset: Dataset[_]): LinearRegressionModel = {

//    // Used to convert untyped dataframes to datasets with vectors
//    implicit val encoder : Encoder[Vector] = ExpressionEncoder()
//
//    val vectors: Dataset[Vector] = dataset.select(dataset($(featuresCol)).as[Vector])
//
//    val dim: Int = AttributeGroup.fromStructField((dataset.schema($(featuresCol)))).numAttributes.getOrElse(
//      vectors.first().size
//    )
//
//    val summary = vectors.rdd.mapPartitions((data: Iterator[Vector]) => {
//      val result = data.foldLeft(new MultivariateOnlineSummarizer())(
//        (summarizer, vector) => summarizer.add(mllib.linalg.Vectors.fromBreeze(vector.asBreeze)))
//      Iterator(result)
//    }).reduce(_ merge _)

    copyValues(new LinearRegressionModel()).setParent(this)

    //    val Row(row: Row) =  dataset
    //      .select(Summarizer.metrics("mean", "std").summary(dataset($(inputCol))))
    //      .first()
    //
    //    copyValues(new LinearRegressionModel(row.getAs[Vector](0).toDense, row.getAs[Vector](1).toDense)).setParent(this)
  }

  override def copy(extra: ParamMap): LinearRegression = defaultCopy(extra)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](override val uid: String)
    extends RegressionModel[Vector, LinearRegressionModel] with LinearRegressionParams with MLWritable {

  private[made] def this() =
    this(Identifiable.randomUID("LinearRegressionModel"))

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(), extra)

  override def predict(features: Vector): Double = {
    0.0
  }

  override def write: MLWriter = new DefaultParamsWriter(this) {
//    override protected def saveImpl(path: String): Unit = {
//      super.saveImpl(path)
//
//      val vectors = means.asInstanceOf[Vector] -> stds.asInstanceOf[Vector]
//
//      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
//    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

//      val vectors = sqlContext.read.parquet(path + "/vectors")

      // Used to convert untyped dataframes to datasets with vectors
//      implicit val encoder : Encoder[Vector] = ExpressionEncoder()

//      val (means, std) =  vectors.select(vectors("_1").as[Vector], vectors("_2").as[Vector]).first()

      val model = new LinearRegressionModel()
      metadata.getAndSetParams(model)
      model
    }
  }
}
