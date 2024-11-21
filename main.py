
from DistributedDbscan import ArgsParser, DebugHelper, IOHelper, DbscanSettings, PartitioningSettings, DistributedDbscan

from pyspark.sql import SparkSession


def main():
    argsParser = ArgsParser()
    argsParser.parse()
    args = argsParser.args

    builder: SparkSession.Builder = SparkSession.builder
    spark = builder \
        .master(args.masterUrl) \
        .appName("DBSCANPP") \
        .config("spark.files", args.pyFiles) \
        .getOrCreate()

    if args.debugOutputPath:
        spark.conf.set(DebugHelper.DebugOutputPath, args.debugOutputPath)

    data = IOHelper.readDataset(spark.sparkContext, args.inputPath)
    settings = DbscanSettings() \
        .withEpsilon(args.eps) \
        .withNumberOfPoints(args.minPts) \
        .withSamplingFraction(args.samplingFraction) \
        .withSamplingStrategy(args.samplingStrategy) \
        .withTreatBorderPointsAsNoise(args.borderPointsAsNoise) \
        .withDistanceMeasure(args.distanceMeasure)

    partitioningSettings = PartitioningSettings(numberOfPointsInBox=args.numberOfPoints)
    clusteringResult = DistributedDbscan.train(data, settings, partitioningSettings)
    IOHelper.saveClusteringResult(clusteringResult, args.outputPath)


if __name__ == '__main__':
    main()
