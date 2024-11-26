from abc import abstractmethod, ABC
from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Iterator, Sequence
from functools import total_ordering
import itertools
import math
from operator import add as op_add
import os
from typing import Optional, Any, Union, Callable, Iterable

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import findspark
findspark.init()

import numpy as np
from pyspark import SparkContext
from pyspark.broadcast import Broadcast
from pyspark.rdd import RDD, portable_hash
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans


class DistanceMeasure(ABC):
    @staticmethod
    @abstractmethod
    def compute(a: Sequence[float], b: Sequence[float]) -> float:
        return 0.0


class EuclideanDistance(DistanceMeasure):
    @staticmethod
    def compute(a: Sequence[float], b: Sequence[float]) -> float:
        return euclidean(a, b)


PointCoordinates = tuple[float, ...]
PointId = int
TempPointId = int
BoxId = int
ClusterId = int
PairOfAdjacentBoxIds = tuple[BoxId, BoxId]

DBSCAN_NOISE_POINT   = 0
DBSCAN_CLUSTER_NEW   = -1
DBSCAN_CLUSTER_UNDEF = -2


@total_ordering
class Point:
    coordinates: PointCoordinates
    pointId: PointId
    boxId: BoxId
    distanceFromOrigin: float
    precomputedNumberOfNeighbors: int
    clusterId: ClusterId
    isSampled: bool

    def __init__(
        self, 
        coordinates: PointCoordinates,
        pointId: PointId = 0,
        boxId: BoxId = 0,
        distanceFromOrigin: float = 0.0,
        precomputedNumberOfNeighbors: int = 0,
        clusterId: ClusterId = DBSCAN_CLUSTER_UNDEF,
        isSampled: bool = True
    ) -> None:
        self.coordinates = coordinates
        self.pointId = pointId
        self.boxId = boxId
        self.distanceFromOrigin = distanceFromOrigin
        self.precomputedNumberOfNeighbors = precomputedNumberOfNeighbors
        self.clusterId = clusterId
        self.isSampled = isSampled

    @classmethod
    def fromArray(cls, coords: Sequence[float]) -> Self:
        return cls(tuple(coords))

    @classmethod
    def fromPoint(cls, pt: Self) -> Self:
        return cls(
            pt.coordinates, pt.pointId, pt.boxId, pt.distanceFromOrigin,
            pt.precomputedNumberOfNeighbors, pt.clusterId, pt.isSampled
        )

    @classmethod
    def fromVariadic(cls, *coords: float) -> Self:
        return cls(tuple(coords))

    def withPointId(self, newId: PointId) -> Self:
        return Point(
            self.coordinates, newId, self.boxId, self.distanceFromOrigin,
            self.precomputedNumberOfNeighbors, self.clusterId, self.isSampled
        )

    def withBoxId(self, newBoxId: BoxId) -> Self:
        return Point(
            self.coordinates, self.pointId, newBoxId, self.distanceFromOrigin,
            self.precomputedNumberOfNeighbors, self.clusterId, self.isSampled
        )

    def withDistanceFromOrigin(self, newDistance: float) -> Self:
        return Point(
            self.coordinates, self.pointId, self.boxId, newDistance,
            self.precomputedNumberOfNeighbors, self.clusterId, self.isSampled
        )

    def withNumberOfNeighbors(self, newNumber: int) -> Self:
        return Point(
            self.coordinates, self.pointId, self.boxId, self.distanceFromOrigin,
            newNumber, self.clusterId, self.isSampled
        )

    def withClusterId(self, newId: ClusterId) -> Self:
        return Point(
            self.coordinates, self.pointId, self.boxId, self.distanceFromOrigin,
            self.precomputedNumberOfNeighbors, newId, self.isSampled
        )

    def withSampling(self, newIsSampled: bool) -> Self:
        return Point(
            self.coordinates, self.pointId, self.boxId, self.distanceFromOrigin,
            self.precomputedNumberOfNeighbors, self.clusterId, newIsSampled
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Point):
            return self.coordinates == other.coordinates
        return False

    def __lt__(self, other):
        if isinstance(other, Point):
            return self.coordinates < other.coordinates
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Point):
            return self.coordinates > other.coordinates
        return NotImplemented

    def __hash__(self) -> int:
        return portable_hash(self.coordinates)

    def __str__(self) -> str:
        return (f"Point at ({', '.join(map(str, self.coordinates))}); id = {self.pointId}; "
                f"box = {self.boxId}; cluster = {self.clusterId}; neighbors = {self.precomputedNumberOfNeighbors}")


RawDataSet = RDD[Point]


class DbscanSettings:
    distanceMeasure: DistanceMeasure
    treatBorderPointsAsNoise: bool
    epsilon: float
    numberOfPoints: int
    samplingFraction: float
    samplingStrategy: str

    # Static
    defaultDistanceMeasure: DistanceMeasure = EuclideanDistance()
    defaultTreatmentOfBorderPoints: bool = False
    defaultEpsilon: float = 1e-4
    defaultNumberOfPoints: int = 3
    defaultSamplingFraction: float = 0.5
    defaultSamplingStrategy: str = 'linspace'
    
    def __init__(self) -> None:
        self.distanceMeasure = self.defaultDistanceMeasure
        self.treatBorderPointsAsNoise = self.defaultTreatmentOfBorderPoints
        self.epsilon = self.defaultEpsilon
        self.numberOfPoints = self.defaultNumberOfPoints
        self.samplingFraction = self.defaultSamplingFraction
        self.samplingStrategy = self.defaultSamplingStrategy

    def withDistanceMeasure(self, dm: DistanceMeasure) -> Self:
        self.distanceMeasure = dm
        return self

    def withTreatBorderPointsAsNoise(self, tbpn: bool) -> Self:
        self.treatBorderPointsAsNoise = tbpn
        return self

    def withEpsilon(self, eps: float) -> Self:
        self.epsilon = eps
        return self

    def withNumberOfPoints(self, np: int) -> Self:
        self.numberOfPoints = np
        return self

    def withSamplingFraction(self, frac: float) -> Self:
        self.samplingFraction = frac
        return self

    def withSamplingStrategy(self, strat: str) -> Self:
        self.samplingStrategy = strat
        return self


class BoundsInOneDimension:
    lower: float
    upper: float
    includeHigherBound: bool

    def __init__(self, lower: float, upper: float, includeHigherBound: bool = False) -> None:
        self.lower = lower
        self.upper = upper
        self.includeHigherBound = includeHigherBound

    def isNumberWithin(self, n: float) -> bool:
        n_ = DoubleComparisonOperations(n)
        return (n_ >= self.lower) and ((n < self.upper) or (self.includeHigherBound and n_ <= self.upper))

    def split(self, n: int, dbscanSettingsOrMinLen: Optional[Union[DbscanSettings, float]] = None) -> list[Self]:
        if dbscanSettingsOrMinLen is not None:
            if isinstance(dbscanSettingsOrMinLen, DbscanSettings):
                minLen = dbscanSettingsOrMinLen.epsilon * 2
            else:
                assert isinstance(dbscanSettingsOrMinLen, (int, float))
                minLen = dbscanSettingsOrMinLen
            maxN = int((self.length / minLen) + 0.5)
            n = min(n, maxN)

        result: list[BoundsInOneDimension] = []
        increment = (self.upper - self.lower) / n
        currentLowerBound = self.lower

        for i in range(1, n + 1):
            include = i == n and self.includeHigherBound
            newUpperBound = currentLowerBound + increment
            newSplit = BoundsInOneDimension(currentLowerBound, newUpperBound, include)
            result.append(newSplit)
            currentLowerBound = newUpperBound

        return result

    @property
    def length(self) -> float:
        return self.upper - self.lower

    def extend(self, byOrByLength: Union[Self, float]) -> Self:
        if isinstance(byOrByLength, BoundsInOneDimension):
            byLength = byOrByLength.length
        else:
            assert isinstance(byOrByLength, (int, float))
            byLength = byOrByLength
        halfLength = byLength / 2.0
        return BoundsInOneDimension(self.lower - halfLength, self.upper + halfLength, self.includeHigherBound)

    def increaseToFit(self, other: Self) -> Self:
        return BoundsInOneDimension(
            min(self.lower, other.lower),
            max(self.upper, other.upper),
            self.includeHigherBound or other.includeHigherBound
        )

    def __str__(self) -> str:
        return f"[{self.lower} - {self.upper}{']' if self.includeHigherBound else ')'}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BoundsInOneDimension):
            return (
                self.lower == other.lower and 
                self.upper == other.upper and 
                self.includeHigherBound == other.includeHigherBound
            )
        return False

    def __hash__(self) -> int:
        return 41 * (41 * (41 + (1 if self.includeHigherBound else 0)) + int(self.lower)) + int(self.upper)

    @classmethod
    def fromTupleOfFloats(cls, tup: tuple[float, float]) -> Self:
        return cls(tup[0], tup[1])

    @classmethod
    def fromTupleOfFloatsAndBool(cls, tup: tuple[float, float, bool]) -> Self:
        return cls(tup[0], tup[1], tup[2])


class DoubleComparisonOperations:
    originalValue: float

    # Static
    Eps: float = 1E-10

    def __init__(self, originalValue: float) -> None:
        self.originalValue = originalValue

    def __eq__(self, other: Any) -> bool:
        return self.isAlmostEqual(self.originalValue, other)

    def __ge__(self, other: Any) -> bool:
        return (self.originalValue > other) or self.isAlmostEqual(self.originalValue, other)

    def __le__(self, other: Any) -> bool:
        return (self.originalValue < other) or self.isAlmostEqual(self.originalValue, other)

    @classmethod
    def isAlmostEqual(cls, x: float, y: float) -> bool:
        return abs(x - y) <= cls.Eps


class BoxIdGenerator:
    initialId: BoxId
    nextId: BoxId

    def __init__(self, initialId: BoxId) -> None:
        self.initialId = initialId
        self.nextId = initialId

    def getNextId(self) -> BoxId:
        self.nextId += 1
        return self.nextId


@total_ordering
class Box:
    bounds: list[BoundsInOneDimension]
    boxId: BoxId
    partitionId: int
    adjacentBoxes: list[Self]
    centerPoint: Point

    def __init__(
        self,
        bounds: list[BoundsInOneDimension],
        boxId: BoxId = 0,
        partitionId: int = -1,
        adjacentBoxes: Optional[list[Self]] = None
    ) -> None:
        self.bounds = bounds
        self.boxId = boxId
        self.partitionId = partitionId
        self.adjacentBoxes = [] if adjacentBoxes is None else adjacentBoxes
        self.centerPoint = self.calculateCenter(bounds)

    @classmethod
    def fromList(cls, bounds: list[BoundsInOneDimension], boxId: BoxId) -> Self:
        return cls(bounds, boxId)

    @classmethod
    def fromBox(cls, box: Self) -> Self:
        return cls(box.bounds, box.boxId, box.partitionId, box.adjacentBoxes)

    @classmethod
    def fromVariadic(cls, *bounds: BoundsInOneDimension) -> Self:
        return cls(list(bounds))

    def splitAlongLongestDimension(self, numberOfSplits: int, idGenerator: Optional[BoxIdGenerator] = None) -> Iterator[Self]:
        longestDimension, idx = self.findLongestDimensionAndItsIndex()

        beforeLongest = self.bounds[:idx]
        afterLongest = self.bounds[idx + 1:]
        splits = longestDimension.split(numberOfSplits)

        if idGenerator is None:
            idGenerator = BoxIdGenerator(self.boxId)

        return (
            Box(beforeLongest + [split] + afterLongest, idGenerator.getNextId())
            for split in splits
        )

    def isPointWithin(self, pt: Point) -> bool:
        assert len(self.bounds) == len(pt.coordinates)
        return all(b.isNumberWithin(c) for b, c in zip(self.bounds, pt.coordinates))

    def isBigEnough(self, settings: DbscanSettings) -> bool:
        return all(b.length >= 2 * settings.epsilon for b in self.bounds)

    def extendBySizeOfOtherBox(self, other: Self) -> Self:
        assert len(self.bounds) == len(other.bounds)
        newBounds = [b1.extend(b2) for b1, b2 in zip(self.bounds, other.bounds)]
        return Box(newBounds)

    def withId(self, newId: BoxId) -> Self:
        return Box(self.bounds, newId, self.partitionId, self.adjacentBoxes)

    def withPartitionId(self, newPartitionId: int) -> Self:
        return Box(self.bounds, self.boxId, newPartitionId, self.adjacentBoxes)

    def __str__(self) -> str:
        return f"Box {', '.join(map(str, self.bounds))}; id = {self.boxId}; partition = {self.partitionId}"

    def findLongestDimensionAndItsIndex(self) -> tuple[BoundsInOneDimension, int]:
        idx: int = 0
        foundBound: Optional[BoundsInOneDimension] = None
        maxLen: float = float('-inf')

        for i, b in enumerate(self.bounds):
            len = b.length
            if len > maxLen:
                foundBound = b
                idx = i
                maxLen = len

        assert foundBound is not None
        return foundBound, idx

    def calculateCenter(self, bounds: list[BoundsInOneDimension]) -> Point:
        centerCoordinates = tuple(b.lower + (b.upper - b.lower) / 2 for b in bounds)
        return Point(centerCoordinates)

    def addAdjacentBox(self, box: Self) -> None:
        self.adjacentBoxes = [box] + self.adjacentBoxes

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Box):
            assert len(self.bounds) == len(other.bounds)
            return self.centerPoint == other.centerPoint
        return False

    def __lt__(self, other: Any):
        if isinstance(other, Box):
            assert len(self.bounds) == len(other.bounds)
            return self.centerPoint < other.centerPoint
        return NotImplemented

    def __gt__(self, other: Any):
        if isinstance(other, Box):
            assert len(self.bounds) == len(other.bounds)
            return self.centerPoint > other.centerPoint
        return NotImplemented

    def isAdjacentToBox(self, other: Self) -> bool:
        assert len(self.bounds) == len(other.bounds)

        hasAdjacentBounds = False
        notAdjacentBounds: list[tuple[BoundsInOneDimension, BoundsInOneDimension]] = []
        for b1, b2 in zip(self.bounds, other.bounds):
            if (
                DoubleComparisonOperations(b1.lower) == b2.lower or
                DoubleComparisonOperations(b1.lower) == b2.upper or
                DoubleComparisonOperations(b1.upper) == b2.upper or
                DoubleComparisonOperations(b1.upper) == b2.lower
            ):
                hasAdjacentBounds = True
            else:
                notAdjacentBounds.append((b1, b2))

        return hasAdjacentBounds and all(
            (
                DoubleComparisonOperations(b1.lower) >= b2.lower and
                DoubleComparisonOperations(b1.upper) <= b2.upper
            ) or (
                DoubleComparisonOperations(b2.lower) >= b1.lower and
                DoubleComparisonOperations(b2.upper) <= b1.upper
            ) for b1, b2 in notAdjacentBounds
        )

    @classmethod
    def apply(cls, centerPoint: Point, size: Self) -> Self:
        newBounds = [BoundsInOneDimension(c, c, True) for c in centerPoint.coordinates]
        return Box(newBounds).extendBySizeOfOtherBox(size)


class DistanceCalculation:
    @classmethod
    def calculatePointDistance(cls, pt1: Point, pt2: Point, distanceMeasure: DistanceMeasure) -> float:
        return cls.calculateDistance(pt1.coordinates, pt2.coordinates, distanceMeasure)

    @staticmethod
    def calculateDistance(pt1: PointCoordinates, pt2: PointCoordinates, distanceMeasure: DistanceMeasure) -> float:
        return distanceMeasure.compute(pt1, pt2)

    @classmethod
    def isPointCloseToAnyBound(cls, pt: Point, box: Box, threshold: float) -> bool:
        return any(cls.isPointCloseToBound(pt, box.bounds[i], i, threshold) for i in range(len(pt.coordinates)))

    @staticmethod
    def isPointCloseToBound(pt: Point, bound: BoundsInOneDimension, dimension: int, threshold: float) -> bool:
        # It will work for Euclidean or Manhattan distance measure but may not work for others
        # TODO: generalize for different distance measures

        x = pt.coordinates[dimension]
        return abs(x - bound.lower) <= threshold or abs(x - bound.upper) <= threshold


class PartitioningSettings:
    numberOfSplits: int
    numberOfLevels: int
    numberOfPointsInBox: int
    numberOfSplitsWithinPartition: int

    # Static
    defaultNumberOfSplitsAlongEachAxis: int = 2
    defaultNumberOfLevels: int = 10
    defaultNumberOfPointsInBox: int = 50000
    defaultNumberOfSplitsWithinPartition: int = 8

    def __init__(
        self,
        numberOfSplits: Optional[int] = None,
        numberOfLevels: Optional[int] = None,
        numberOfPointsInBox: Optional[int] = None,
        numberOfSplitsWithinPartition: Optional[int] = None
    ) -> None:
        self.numberOfSplits = self.defaultNumberOfSplitsAlongEachAxis if numberOfSplits is None else numberOfSplits
        self.numberOfLevels = self.defaultNumberOfLevels if numberOfLevels is None else numberOfLevels
        self.numberOfPointsInBox = self.defaultNumberOfPointsInBox if numberOfPointsInBox is None else numberOfPointsInBox
        self.numberOfSplitsWithinPartition = self.defaultNumberOfSplitsWithinPartition if numberOfSplitsWithinPartition is None else numberOfSplitsWithinPartition

    def withNumberOfLevels(self, nl: int) -> Self:
        return PartitioningSettings(self.numberOfSplits, nl, self.numberOfPointsInBox, self.numberOfSplitsWithinPartition)

    def withNumberOfSplitsWithinPartition(self, ns: int) -> Self:
        return PartitioningSettings(self.numberOfSplits, self.numberOfLevels, self.numberOfPointsInBox, ns)


@total_ordering
class PointSortKey:
    boxId: BoxId
    pointId: PointId

    def __init__(self, pt: Point) -> None:
        self.boxId = pt.boxId
        self.pointId = pt.pointId

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, PointSortKey) and
            self.boxId == other.boxId and
            self.pointId == other.pointId
        )

    def __lt__(self, other: Any):
        if isinstance(other, PointSortKey):
            if self.boxId != other.boxId:
                return self.boxId < other.boxId
            return self.pointId < other.pointId
        return NotImplemented

    def __gt__(self, other: Any):
        if isinstance(other, PointSortKey):
            if self.boxId != other.boxId:
                return self.boxId > other.boxId
            return self.pointId > other.pointId
        return NotImplemented

    def __hash__(self) -> int:
        return 41 * (41 * self.pointId) + self.boxId

    def __str__(self) -> str:
        return f"PointSortKey with box: {self.boxId} , ptId: {self.pointId}"


class BoxPartitioner:
    boxes: Sequence[Box]
    boxIdsToPartitions: dict[BoxId, int]

    def __init__(self, boxes: Sequence[Box]) -> None:
        self.boxes = boxes
        assert all(box.partitionId >= 0 for box in boxes)
        self.boxIdsToPartitions = self._generateBoxIdsToPartitionsMap(boxes)

    @property
    def numPartitions(self) -> int:
        return len(self.boxes)

    def getPartition(self, key: Any) -> int:
        if isinstance(key, (Point, PointSortKey)):
            return self.boxIdsToPartitions[key.boxId]
        elif isinstance(key, BoxId):
            return self.boxIdsToPartitions[key]
        else:
            raise KeyError(f'Excepted key of type {Point}, {PointSortKey}, or {BoxId}, got {type(key)}')

    @staticmethod
    def _generateBoxIdsToPartitionsMap(boxes: Sequence[Box]) -> dict[BoxId, int]:
        return {box.boxId: box.partitionId for box in boxes}

    @staticmethod
    def assignPartitionIdsToBoxes(boxes: Sequence[Box]) -> tuple[Box, ...]:
        return tuple(box.withPartitionId(i) for i, box in enumerate(boxes))


class BoxTreeItemBase:
    box: Box
    children: list[Self]

    def __init__(self, box: Box) -> None:
        self.box = box
        self.children = []

    @property
    def flatten(self) -> list[Self]:
        return [self] + [item for child in self.children for item in child.flatten]

    @property
    def flattenBoxes(self) -> list[Box]:
        return [item.box for item in self.flatten]

    def flattenBoxesWithPredicate(self, predicate: Callable[[Self], bool]) -> list[Box]:
        result: list[Box] = []
        self._flattenBoxes(predicate, result)
        return result

    def _flattenBoxes(self, predicate: Callable[[Self], bool], buffer: list[Box]) -> None:
        if self.children and any(predicate(child) for child in self.children):
            for child in self.children:
                child._flattenBoxes(predicate, buffer)
        else:
            buffer.append(self.box)


class BoxTreeItemWithNumberOfPoints(BoxTreeItemBase):
    numberOfPoints: int

    def __init__(self, b: Box) -> None:
        super().__init__(b)
        self.numberOfPoints = 0

    def clone(self) -> Self:
        result = BoxTreeItemWithNumberOfPoints(self.box)
        result.children = [child.clone() for child in self.children]
        return result


class BoxCalculator:
    data: RawDataSet
    numberOfDimensions: int

    def __init__(self, data: RawDataSet) -> None:
        self.data = data
        self.numberOfDimensions = self._getNumberOfDimensions(data)

    def generateDensityBasedBoxes(
        self,
        partitioningSettings: Optional[PartitioningSettings] = None,
        dbscanSettings: Optional[DbscanSettings] = None
    ) -> tuple[Sequence[Box], Box]:
        if partitioningSettings is None:
            partitioningSettings = PartitioningSettings()
        if dbscanSettings is None:
            dbscanSettings = DbscanSettings()

        rootBox = self.calculateBoundingBox
        boxTree = self.generateTreeOfBoxes(rootBox, partitioningSettings, dbscanSettings)

        broadcastBoxTree = self.data.context.broadcast(boxTree)
        f_countPointsInOnePartition = self.countPointsInOnePartition

        partialCounts: RDD[tuple[BoxId, int]] = self.data.mapPartitions(
            lambda it: f_countPointsInOnePartition(broadcastBoxTree.value.clone(), it)
        )

        totalCounts = partialCounts.reduceByKeyLocally(op_add)
        numberOfPointsInBox = partitioningSettings.numberOfPointsInBox
        boxesWithEnoughPoints = boxTree.flattenBoxesWithPredicate(
            lambda x: totalCounts[x.box.boxId] >= numberOfPointsInBox
        )

        self._assignAdjacentBoxes(boxesWithEnoughPoints)

        return BoxPartitioner.assignPartitionIdsToBoxes(boxesWithEnoughPoints), rootBox

    @staticmethod
    def _getNumberOfDimensions(ds: RawDataSet) -> int:
        pt = ds.first()
        return len(pt.coordinates)

    @property
    def calculateBoundingBox(self) -> Box:
        return Box(self._calculateBounds(self.data, self.numberOfDimensions))

    @classmethod
    def _calculateBounds(cls, ds: RawDataSet, dimensions: int) -> list[BoundsInOneDimension]:
        minPoint = Point((float( 'inf'),) * dimensions)
        maxPoint = Point((float('-inf'),) * dimensions)

        mins = cls._fold(ds, minPoint, lambda x: min(x[0], x[1]))
        maxs = cls._fold(ds, maxPoint, lambda x: max(x[0], x[1]))

        return [
            BoundsInOneDimension(minCoord, maxCoord, True)
            for minCoord, maxCoord in zip(mins.coordinates, maxs.coordinates)
        ]

    @staticmethod
    def _fold(ds: RawDataSet, zeroValue: Point, mapFunction: Callable[[tuple[float, float]], float]) -> Point:
        return ds.fold(zeroValue, lambda pt1, pt2: Point(tuple(
            mapFunction(pair) for pair in zip(pt1.coordinates, pt2.coordinates)
        )))

    @classmethod
    def generateTreeOfBoxes(
        cls,
        root: Box,
        partitioningSettings: PartitioningSettings,
        dbscanSettings: DbscanSettings,
        idGenerator: Optional[BoxIdGenerator] = None
    ) -> BoxTreeItemWithNumberOfPoints:
        if idGenerator is None:
            idGenerator = BoxIdGenerator(root.boxId)

        result = BoxTreeItemWithNumberOfPoints(root)
        if partitioningSettings.numberOfLevels > 0:
            newPartitioningSettings = partitioningSettings.withNumberOfLevels(partitioningSettings.numberOfLevels-1)

            result.children = [
                cls.generateTreeOfBoxes(child, newPartitioningSettings, dbscanSettings, idGenerator)
                for child in root.splitAlongLongestDimension(partitioningSettings.numberOfSplits, idGenerator)
                if child.isBigEnough(dbscanSettings)
            ]

        return result

    @classmethod
    def countOnePoint(cls, pt: Point, root: BoxTreeItemWithNumberOfPoints) -> None:
        if root.box.isPointWithin(pt):
            root.numberOfPoints += 1
            for child in root.children:
                cls.countOnePoint(pt, child)

    @classmethod
    def countPointsInOnePartition(cls, root: BoxTreeItemWithNumberOfPoints, it: Iterable[Point]) -> Iterator[tuple[BoxId, int]]:
        for pt in it:
            cls.countOnePoint(pt, root)

        return ((x.box.boxId, x.numberOfPoints) for x in root.flatten)

    @classmethod
    def _generateCombinationsOfSplits(
        cls,
        splits: list[list[BoundsInOneDimension]],
        dimensionIndex: int
    ) -> list[list[BoundsInOneDimension]]:
        if dimensionIndex < 0:
            return [[]]
        else:
            combinations = cls._generateCombinationsOfSplits(splits, dimensionIndex - 1)
            return[[j] + i for i in combinations for j in splits[dimensionIndex]]

    @classmethod
    def splitBoxIntoEqualBoxes(cls, rootBox: Box, maxSplits: int, dbscanSettings: DbscanSettings) -> list[Box]:
        dimensions = len(rootBox.bounds)
        splits = [bound.split(maxSplits, dbscanSettings) for bound in rootBox.bounds]
        combinations = cls._generateCombinationsOfSplits(splits, dimensions - 1)

        return [Box(combination[::-1], i + 1) for i, combination in enumerate(combinations)]
    
    @staticmethod
    def _assignAdjacentBoxes(boxesWithEnoughPoints: Sequence[Box]) -> None:
        temp = boxesWithEnoughPoints
        for i in range(len(temp)):
            for j in range(i + 1, len(temp)):
                if temp[i].isAdjacentToBox(temp[j]):
                    temp[i].addAdjacentBox(temp[j])
                    temp[j].addAdjacentBox(temp[i])

    @staticmethod
    def generateDistinctPairsOfAdjacentBoxIds(boxesWithAdjacentBoxes: Iterable[Box]) -> Iterator[PairOfAdjacentBoxIds]:
        return ((b.boxId, ab.boxId) for b in boxesWithAdjacentBoxes for ab in b.adjacentBoxes if b.boxId < ab.boxId)

    @staticmethod
    def _shouldAdjacentBoxBeIncludedInPartition(rootBoxId: BoxId, adjacentBoxId: BoxId) -> bool:
        return rootBoxId <= adjacentBoxId

    @staticmethod
    def _generateEmbracingBox(boxes: Iterable[Box]) -> Box:
        it = iter(boxes)
        firstBox = next(it)
        embracingBoxBounds = firstBox.bounds

        for b in it:
            assert len(embracingBoxBounds) == len(b.bounds)
            embracingBoxBounds = [
                x.increaseToFit(y) for x, y in zip(embracingBoxBounds, b.bounds)
            ]

        return Box(embracingBoxBounds)

    @classmethod
    def _generateEmbracingBoxFromAdjacentBoxes(cls, rootBox: Box) -> Box:
        rootAndAdjacentBoxes = [rootBox] + [
            x for x in rootBox.adjacentBoxes
            if cls._shouldAdjacentBoxBeIncludedInPartition(rootBox.boxId, x.boxId)
        ]
        return cls._generateEmbracingBox(rootAndAdjacentBoxes)


class PointIndexer:
    numberOfPartitions: int
    currentPartition: int
    multiplier: int
    currentIndex: int

    def __init__(self, numberOfPartitions: int, currentPartition: int) -> None:
        self.numberOfPartitions = numberOfPartitions
        self.currentPartition = currentPartition
        self.multiplier = self.computeMultiplier(numberOfPartitions)
        self.currentIndex = 0

    @property
    def getNextIndex(self) -> int:
        self.currentIndex += 1
        return self.currentIndex * self.multiplier + self.currentPartition

    @staticmethod
    def computeMultiplier(numberOfPartitions: int) -> int:
        numberOfDigits = math.floor(math.log10(numberOfPartitions)) + 1
        return round(math.pow(10, numberOfDigits))

    @classmethod
    def addMetadataToPoints(
        cls,
        data: RawDataSet,
        boxes: Broadcast[Iterable[Box]],
        dimensions: Broadcast[int],
        distanceMeasure: DistanceMeasure,
    ) -> RDD[tuple[PointSortKey, Point]]:
        numPartitions = data.getNumPartitions()
        origin = Point((0.0,) * dimensions.value)

        def f_mapPartitionsWithIndex(partitionIndex: int, points: Iterable[Point]) -> list[tuple[PointSortKey, Point]]:
            boxes_ = boxes.value
            pointIndexer = cls(numPartitions, partitionIndex)
            result = []
            for pt in points:
                pointIndex = pointIndexer.getNextIndex
                box = next((b for b in boxes_ if b.isPointWithin(pt)), None)
                distanceFromOrigin = distanceMeasure.compute(pt.coordinates, origin.coordinates)
                if box is None:
                    raise RuntimeError(f"Box for point {pt} was not found (metadata)")
                boxId = box.boxId
                newPoint = Point(
                    pt.coordinates,
                    pointIndex,
                    boxId,
                    distanceFromOrigin,
                    pt.precomputedNumberOfNeighbors,
                    pt.clusterId,
                    pt.isSampled,
                )
                result.append((PointSortKey(newPoint), newPoint))
            return result

        return data.mapPartitionsWithIndex(f_mapPartitionsWithIndex)


class PointsPartitionedByBoxesRDD:
    boxes: Sequence[Box]
    boundingBox: Box
    rdd: RDD[tuple[PointSortKey, Point]]

    def __init__(self, prev: RDD[tuple[PointSortKey, Point]], boxes: Sequence[Box], boundingBox: Box) -> None:
        self.boxes = boxes
        self.boundingBox = boundingBox
        partitioner = BoxPartitioner(boxes)
        self.rdd = prev.partitionBy(partitioner.numPartitions, partitioner.getPartition)

    @classmethod
    def apply(
        cls,
        rawData: RawDataSet,
        partitioningSettings: Optional[PartitioningSettings] = None,
        dbscanSettings: Optional[DbscanSettings] = None
    ) -> Self:
        if partitioningSettings is None:
            partitioningSettings = PartitioningSettings()
        if dbscanSettings is None:
            dbscanSettings = DbscanSettings()

        sc = rawData.context
        boxCalculator = BoxCalculator(rawData)
        boxes, boundingBox = boxCalculator.generateDensityBasedBoxes(partitioningSettings, dbscanSettings)
        broadcastBoxes = sc.broadcast(boxes)
        broadcastNumberOfDimensions = sc.broadcast(boxCalculator.numberOfDimensions)

        pointsInBoxes = PointIndexer.addMetadataToPoints(
            rawData,
            broadcastBoxes,
            broadcastNumberOfDimensions,
            dbscanSettings.distanceMeasure
        )

        return cls(pointsInBoxes, boxes, boundingBox)

    @staticmethod
    def extractPointIdsAndCoordinates(data: RDD[tuple[PointSortKey, Point]]) -> RDD[tuple[PointId, PointCoordinates]]:
        return data.map(lambda x: (x[1].pointId, x[1].coordinates))


class DebugHelper:
    # Static
    DebugOutputPath = "debug.output.path"

    @classmethod
    def doAndSaveResult(cls, sc: SparkContext, relativePath: str, fn: Callable[[str], None]) -> None:
        opt = sc.getConf().get(cls.DebugOutputPath, None)
        if opt is not None:
            fn(os.path.join(opt, relativePath))

    @classmethod
    def justDo(cls, sc: SparkContext, fn: Callable[[], None]) -> None:
        opt = sc.getConf().get(cls.DebugOutputPath, None)
        if opt is not None:
            fn()


class BoxTreeItemWithPoints(BoxTreeItemBase):
    # Original code uses a synchronized array buffer, but we assume this isn't an issue in Python/PySpark
    points: list[Point]
    adjacentBoxes: list[Self]

    def __init__(
        self,
        b: Box,
        points: Optional[list[Point]] = None,
        adjacentBoxes: Optional[list[Self]] = None
    ) -> None:
        super().__init__(b)
        self.points = [] if points is None else points
        self.adjacentBoxes = [] if adjacentBoxes is None else adjacentBoxes


class PartitionIndex(DistanceCalculation):
    partitionBounds: Box
    dbscanSettings: DbscanSettings
    partitioningSettings: PartitioningSettings
    distanceMeasure: DistanceMeasure
    _boxesTree: BoxTreeItemWithPoints
    _largeBox: Box

    def __init__(self, partitionBounds: Box, dbscanSettings: DbscanSettings, partitioningSettings: PartitioningSettings) -> None:
        self.partitionBounds = partitionBounds
        self.dbscanSettings = dbscanSettings
        self.partitioningSettings = partitioningSettings
        self.distanceMeasure = dbscanSettings.distanceMeasure
        self._boxesTree = self.buildTree(
            partitionBounds, partitioningSettings, dbscanSettings
        )
        self._largeBox = self.createBoxTwiceLargerThanLeaf(self._boxesTree)

    def populate(self, points: Iterable[Point]) -> None:
        # clock = Clock()

        for pt in points:
            self._findBoxAndAddPoint(pt, self._boxesTree)

        # clock.log_time_since_start("Population of partition index")

    def findClosePoints(self, pt: Point) -> Iterator[Point]:
        return (p for p in self._findPotentiallyClosePoints(pt) if p.pointId != pt.pointId and self.calculatePointDistance(p, pt, self.distanceMeasure) <= self.dbscanSettings.epsilon)

    def _findPotentiallyClosePoints(self, pt: Point) -> list[Point]:
        box1 = self._findBoxForPoint(pt, self._boxesTree)
        result = [p for p in box1.points if p.pointId != pt.pointId and abs(p.distanceFromOrigin - pt.distanceFromOrigin) <= self.dbscanSettings.epsilon]

        if self.isPointCloseToAnyBound(pt, box1.box, self.dbscanSettings.epsilon):
            for box2 in box1.adjacentBoxes:
                tempBox = Box.apply(pt, self._largeBox)
                if tempBox.isPointWithin(box2.box.centerPoint):
                    result.extend([p for p in box2.points if abs(p.distanceFromOrigin - pt.distanceFromOrigin) <= self.dbscanSettings.epsilon])

        return result

    def _findBoxAndAddPoint(self, pt: Point, root: BoxTreeItemWithPoints) -> None:
        b = self._findBoxForPoint(pt, root)
        b.points.append(pt)

    def _findBoxForPoint(self, pt: Point, root: BoxTreeItemWithPoints) -> BoxTreeItemWithPoints:
        if not root.children:
            return root
        child = next((x for x in root.children if x.box.isPointWithin(pt)), None)
        if child is not None:
            return self._findBoxForPoint(pt, child)
        else:
            raise RuntimeError(f"Box for point {pt} was not found")

    @classmethod
    def buildTree(
        cls,
        boundingBox: Box,
        partitioningSettings: PartitioningSettings,
        dbscanSettings: DbscanSettings
    ) -> BoxTreeItemWithPoints:
        sortedBoxes = cls.generateAndSortBoxes(
            boundingBox, partitioningSettings.numberOfSplitsWithinPartition, dbscanSettings
        )
        return cls._buildTree(boundingBox, sortedBoxes)

    @classmethod
    def _buildTree(cls, boundingBox: Box, sortedBoxes: list[Box]) -> BoxTreeItemWithPoints:
        leafs = [BoxTreeItemWithPoints(b) for b in sortedBoxes]

        for leaf in leafs:
            leaf.adjacentBoxes.extend(cls.findAdjacentBoxes(leaf, leafs))

        root = BoxTreeItemWithPoints(boundingBox)
        root.children = cls.generateSubitems(root, 0, leafs, 0, len(leafs) - 1)
        return root

    @classmethod
    def generateSubitems(cls, root: BoxTreeItemWithPoints, dimension: int, leafs: list[BoxTreeItemWithPoints], start: int, end: int) -> list[BoxTreeItemWithPoints]:
        numberOfDimensions = len(root.box.bounds)
        result: list[BoxTreeItemWithPoints] = []

        if dimension < numberOfDimensions:
            nodeStart = start
            nodeEnd = start

            while nodeStart <= end:
                b = leafs[nodeStart].box.bounds[dimension]
                leafsSubset: list[BoxTreeItemWithPoints] = []

                while nodeEnd <= end and leafs[nodeEnd].box.bounds[dimension] == b:
                    leafsSubset.append(leafs[nodeEnd])
                    nodeEnd += 1

                if len(leafsSubset) > 1:
                    embracingBox = cls.generateEmbracingBox(leafsSubset, numberOfDimensions)
                    newSubitem = BoxTreeItemWithPoints(embracingBox)
                    newSubitem.children = cls.generateSubitems(
                        newSubitem, dimension + 1, leafs, nodeStart, nodeEnd - 1
                    )
                else:
                    newSubitem = leafsSubset[0]

                result.append(newSubitem)
                nodeStart = nodeEnd

        return result

    @staticmethod
    def generateEmbracingBox(subitems: Iterable[BoxTreeItemWithPoints], numberOfDimensions: int) -> Box:
        dimensions: list[BoundsInOneDimension] = []

        for i in range(numberOfDimensions):
            zeroValue = BoundsInOneDimension(float('inf'), float('-inf'), False)
            x = [item.box.bounds[i] for item in subitems]
            newDimension = zeroValue
            for dim in x:
                newDimension = BoundsInOneDimension(
                    min(newDimension.lower, dim.lower),
                    max(newDimension.upper, dim.upper),
                    newDimension.includeHigherBound or dim.includeHigherBound,
                )
            dimensions.append(newDimension)

        return Box(dimensions)

    @staticmethod
    def generateAndSortBoxes(boundingBox: Box, maxNumberOfSplits: int, dbscanSettings: DbscanSettings) -> list[Box]:
        result = BoxCalculator.splitBoxIntoEqualBoxes(boundingBox, maxNumberOfSplits, dbscanSettings)
        result.sort()
        return result

    @staticmethod
    def findAdjacentBoxes(x: BoxTreeItemWithPoints, boxes: Iterable[BoxTreeItemWithPoints]) -> Iterator[BoxTreeItemWithPoints]:
        result = []

        for y in boxes:
            if y == x:
                continue
            n = 0
            for i in range(len(x.box.bounds)):
                cx = x.box.centerPoint.coordinates[i]
                cy = y.box.centerPoint.coordinates[i]
                d = DoubleComparisonOperations(abs(cx - cy))
                if d == 0 or d == x.box.bounds[i].length:
                    n += 1
            if n == len(x.box.bounds):
                result.append(y)

        return reversed(result)

    @classmethod
    def createBoxTwiceLargerThanLeaf(cls, root: BoxTreeItemWithPoints) -> Box:
        leaf = cls.findFirstLeafBox(root)
        return leaf.extendBySizeOfOtherBox(leaf)

    @staticmethod
    def findFirstLeafBox(root: BoxTreeItemWithPoints) -> Box:
        result = root
        while result.children:
            result = result.children[0]
        return result.box


class AdjacentBoxesPartitioner:
    adjacentBoxIdPairs: list[PairOfAdjacentBoxIds]

    def __init__(self, adjacentBoxIdPairs: list[PairOfAdjacentBoxIds]) -> None:
        self.adjacentBoxIdPairs = adjacentBoxIdPairs

    @classmethod
    def fromBoxes(cls, boxesWithAdjacentBoxes: Iterable[Box]):
        return cls(list(BoxCalculator.generateDistinctPairsOfAdjacentBoxIds(boxesWithAdjacentBoxes)))

    @property
    def numPartitions(self) -> int:
        return len(self.adjacentBoxIdPairs)

    def getPartition(self, key: Any) -> int:
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], BoxId) and isinstance(key[1], BoxId):
            return self.adjacentBoxIdPairs.index(key)
        else:
            raise KeyError(f'Excepted key as tuple of two {BoxId} instances, got {type(key)}')


class PointsInAdjacentBoxesRDD:
    adjacentBoxIdPairs: list[PairOfAdjacentBoxIds]
    rdd: RDD[tuple[PairOfAdjacentBoxIds, Point]]

    def __init__(self, prev: RDD[tuple[PairOfAdjacentBoxIds, Point]], adjacentBoxIdPairs: list[PairOfAdjacentBoxIds]) -> None:
        self.adjacentBoxIdPairs = adjacentBoxIdPairs
        partitioner = AdjacentBoxesPartitioner(adjacentBoxIdPairs)
        self.rdd = prev.partitionBy(partitioner.numPartitions, partitioner.getPartition)

    @classmethod
    def apply(cls, points: RDD[Point], boxesWithAdjacentBoxes: Iterable[Box]) -> Self:
        adjacentBoxIdPairs = list(BoxCalculator.generateDistinctPairsOfAdjacentBoxIds(boxesWithAdjacentBoxes))

        broadcastBoxIdPairs = points.context.broadcast(adjacentBoxIdPairs)

        def f_mapPartitions(it: Iterable[Point]) -> Iterator[tuple[PairOfAdjacentBoxIds, Point]]:
            boxIdPairs = broadcastBoxIdPairs.value
            return (
                (pair, p)
                for p in it
                for pair in boxIdPairs
                if p.boxId == pair[0] or p.boxId == pair[1]
            )

        pointsKeyedByPairOfBoxes = points.mapPartitions(f_mapPartitions)

        return cls(pointsKeyedByPairOfBoxes, adjacentBoxIdPairs)


class DistanceAnalyzer(DistanceCalculation):
    settings: DbscanSettings
    partitioningSettings: PartitioningSettings
    distanceMeasure: DistanceMeasure

    def __init__(self, settings: Optional[DbscanSettings] = None, partitioningSettings: Optional[PartitioningSettings] = None) -> None:
        self.settings = DbscanSettings() if settings is None else settings
        self.partitioningSettings = PartitioningSettings() if partitioningSettings is None else partitioningSettings
        self.distanceMeasure = self.settings.distanceMeasure

    def countNeighborsForEachPoint(self, data: PointsPartitionedByBoxesRDD) -> RDD[tuple[PointSortKey, Point]]:
        closePointCounts = self.countClosePoints(data) \
            .reduceByKey(op_add) \
            .mapValues(lambda count: (count + 1)) \
            .cache()

        pointsWithoutNeighbors = data.rdd.keys().subtract(closePointCounts.keys()).map(lambda x: (x, 1))

        allPointCounts = closePointCounts.union(pointsWithoutNeighbors)

        closePointCounts.unpersist()
        del closePointCounts

        partitioner = BoxPartitioner(data.boxes)
        partitionedAndSortedCounts = (
            allPointCounts
            .repartitionAndSortWithinPartitions(partitioner.numPartitions, partitioner.getPartition, ascending=True)
        )

        sortedData: RDD[tuple[PointSortKey, Point]] = data.rdd.mapPartitions(lambda it: sorted(it, key=lambda x: x[0]), preservesPartitioning=True)

        def f_map(x: tuple[tuple[PointSortKey, Point], tuple[PointSortKey, int]]) -> tuple[PointSortKey, Point]:
            assert x[0][0].pointId == x[1][0].pointId
            newPt = x[0][1].withNumberOfNeighbors(x[1][1])
            return PointSortKey(newPt), newPt

        pointsWithCounts = sortedData.zip(partitionedAndSortedCounts).map(f_map)

        def f_saveResult(path: str) -> None:
            pointsWithCounts.mapPartitionsWithIndex(
                lambda idx, it: (
                    f"{','.join(map(str, x[1].coordinates))},{idx}" for x in it
                )
            ).saveAsTextFile(path)

        DebugHelper.doAndSaveResult(
            data.rdd.context,
            "restoredPartitions",
            f_saveResult
        )

        return pointsWithCounts

    def countClosePoints(self, data: PointsPartitionedByBoxesRDD) -> RDD[tuple[PointSortKey, int]]:
        closePointsInsideBoxes = self.countClosePointsWithinEachBox(data)
        pointsCloseToBoxBounds = self.findPointsCloseToBoxBounds(data.rdd, data.boxes, self.settings.epsilon)
        closePointsInDifferentBoxes = self._countClosePointsInDifferentBoxes(
            pointsCloseToBoxBounds, data.boxes, self.settings.epsilon
        )

        return closePointsInsideBoxes.union(closePointsInDifferentBoxes)

    def countClosePointsWithinEachBox(self, data: PointsPartitionedByBoxesRDD) -> RDD[tuple[PointSortKey, int]]:
        broadcastBoxes = data.rdd.context.broadcast(data.boxes)
        f_countClosePointsWithinPartition = self._countClosePointsWithinPartition
        settings = self.settings
        partitioningSettings = self.partitioningSettings

        def f_mapPartitions(partitionIndex: int, it: Iterable[tuple[PointSortKey, Point]]) -> Iterable[tuple[PointSortKey, int]]:
            boxes = broadcastBoxes.value
            boundingBox = next(box for box in boxes if box.partitionId == partitionIndex)
            return f_countClosePointsWithinPartition(it, boundingBox, settings, partitioningSettings)

        return data.rdd.mapPartitionsWithIndex(f_mapPartitions)

    @staticmethod
    def _countClosePointsWithinPartition(it: Iterable[tuple[PointSortKey, Point]], boundingBox: Box, settings: DbscanSettings, partitioningSettings: PartitioningSettings) -> Iterable[tuple[PointSortKey, int]]:
        it1, it2 = itertools.tee(it)

        partitionIndex = PartitionIndex(boundingBox, settings, partitioningSettings)
        counts: defaultdict[PointSortKey, int] = defaultdict(int)

        partitionIndex.populate(pt[1] for pt in it1)

        for currentPoint in it2:
            if currentPoint[1].isSampled:
                closePointsCount = sum(1 for _ in partitionIndex.findClosePoints(currentPoint[1]))
                counts[currentPoint[0]] += closePointsCount

        return counts.items()

    def findPointsCloseToBoxBounds(self, data: RDD[tuple[PointSortKey, Point]], boxes: Iterable[Box], eps: float) -> RDD[Point]:
        broadcastBoxes = data.context.broadcast(boxes)
        f_isPointCloseToAnyBound = self.isPointCloseToAnyBound

        def f_mapPartitions(index: int, it: Iterable[tuple[PointSortKey, Point]]) -> Iterable[Point]:
            boxes_ = broadcastBoxes.value
            box = next(box for box in boxes_ if box.partitionId == index)
            return (pt[1] for pt in it if f_isPointCloseToAnyBound(pt[1], box, eps))

        return data.mapPartitionsWithIndex(f_mapPartitions)

    def _countClosePointsInDifferentBoxes(self, data: RDD[Point], boxesWithAdjacentBoxes: Iterable[Box], eps: float) -> RDD[tuple[PointSortKey, int]]:
        pointsInAdjacentBoxes = PointsInAdjacentBoxesRDD.apply(data, boxesWithAdjacentBoxes).rdd

        self_eps = self.settings.epsilon
        distanceMeasure = self.distanceMeasure
        f_calculatePointDistance = self.calculatePointDistance

        def f_mapPartitions(idx: int, it: Iterable[tuple[PairOfAdjacentBoxIds, Point]]) -> Iterable[tuple[PointSortKey, int]]:
            pointsInPartition = sorted((pt[1] for pt in it), key=lambda pt: pt.distanceFromOrigin)
            counts: defaultdict[PointSortKey, int] = defaultdict(int)

            for i in range(1, len(pointsInPartition)):
                pi = pointsInPartition[i]
                piIsSampled = pi.isSampled
                piSortKey = PointSortKey(pi)
                for j in range(i - 1, -1, -1):
                    pj = pointsInPartition[j]
                    if pi.distanceFromOrigin - pj.distanceFromOrigin > eps:
                        break

                    pjIsSampled = pj.isSampled
                    if (piIsSampled or pjIsSampled) and pi.boxId != pj.boxId and f_calculatePointDistance(pi, pj, distanceMeasure) <= self_eps:
                        if piIsSampled: counts[piSortKey] += 1
                        if pjIsSampled: counts[PointSortKey(pj)] += 1

            return counts.items()

        return pointsInAdjacentBoxes.mapPartitionsWithIndex(f_mapPartitions)

    def findNeighborsOfNewPoint(self, clusteredAndNoisePoints: RDD[Point], newPoint: PointCoordinates) -> RDD[Point]:
        eps = self.settings.epsilon
        distanceMeasure = self.distanceMeasure
        f_calculateDistance = self.calculateDistance
        return clusteredAndNoisePoints.filter(lambda pt: (f_calculateDistance(pt.coordinates, newPoint, distanceMeasure) <= eps))


class DbscanModel:
    allPoints: RDD[Point]
    settings: DbscanSettings

    def __init__(self, allPoints: RDD[Point], settings: DbscanSettings) -> None:
        self.allPoints = allPoints
        self.settings = settings

    def predict(self, newPoint: Point) -> ClusterId:
        distanceAnalyzer = DistanceAnalyzer(self.settings)
        neighborCountsByCluster = distanceAnalyzer.findNeighborsOfNewPoint(self.allPoints, newPoint.coordinates) \
            .map(lambda x: (x.clusterId, x)) \
            .countByKey()

        neighborCountsWithoutNoise = {k: v for k, v in neighborCountsByCluster.items() if k != DBSCAN_NOISE_POINT}
        possibleClusters = {k: v for k, v in neighborCountsWithoutNoise.items() if v >= self.settings.numberOfPoints - 1}
        noisePointsCount = neighborCountsByCluster.get(DBSCAN_NOISE_POINT, 0)

        if len(possibleClusters) >= 1:
            return next(iter(possibleClusters))

        elif len(neighborCountsWithoutNoise) >= 1 and not self.settings.treatBorderPointsAsNoise:
            return next(iter(neighborCountsWithoutNoise))

        elif noisePointsCount >= self.settings.numberOfPoints - 1:
            return DBSCAN_CLUSTER_NEW

        else:
            return DBSCAN_NOISE_POINT

    def noisePoints(self) -> RDD[Point]:
        return self.allPoints.filter(lambda point: point.clusterId == DBSCAN_NOISE_POINT)

    def clusteredPoints(self) -> RDD[Point]:
        return self.allPoints.filter(lambda point: point.clusterId != DBSCAN_NOISE_POINT)


class Args:
    # CommonArgs
    masterUrl: str
    pyFiles: str
    inputPath: str
    outputPath: str
    distanceMeasure: DistanceMeasure
    debugOutputPath: Optional[str]
    # EpsArg
    eps: float
    # NumberOfPointsInPartitionArg
    numberOfPoints: int
    # Args
    minPts: int
    borderPointsAsNoise: bool
    # DBSCAN++
    samplingFraction: float
    samplingStrategy: str
    
    def __init__(
        self,
        *,
        masterUrl: str = '',
        pyFiles: str = '',
        inputPath: str = '',
        outputPath: str = '',
        distanceMeasure: DistanceMeasure = DbscanSettings.defaultDistanceMeasure,
        debugOutputPath: Optional[str] = None,
        eps: float = DbscanSettings.defaultEpsilon,
        numberOfPoints: int = PartitioningSettings.defaultNumberOfPointsInBox,
        minPts: int = DbscanSettings.defaultNumberOfPoints,
        borderPointsAsNoise: bool = DbscanSettings.defaultTreatmentOfBorderPoints,
        samplingFraction: float = DbscanSettings.defaultSamplingFraction,
        samplingStrategy: str = DbscanSettings.defaultSamplingStrategy
    ) -> None:
        self.masterUrl = masterUrl
        self.pyFiles = pyFiles
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.distanceMeasure = distanceMeasure
        self.debugOutputPath = debugOutputPath
        self.eps = eps
        self.numberOfPoints = numberOfPoints
        self.minPts = minPts
        self.borderPointsAsNoise = borderPointsAsNoise
        self.samplingFraction = samplingFraction
        self.samplingStrategy = samplingStrategy


class ArgsParser:
    args: Args
    _parser: ArgumentParser

    def __init__(self) -> None:
        self.args = Args()
        self._parser = ArgumentParser(prog="DBSCAN clustering algorithm")

        # Common arguments
        self._parser.add_argument('--ds-master', required=True, metavar='<url>', help='Master URL')
        self._parser.add_argument('--ds-pyFiles', required=True, metavar='<pyFiles>', help='Path to DBSCAN implementation files to make visible to all nodes in your cluster')
        self._parser.add_argument('--ds-input', required=True, metavar='<path>', help='Input path')
        self._parser.add_argument('--ds-output', required=True, metavar='<path>', help='Output path')
        self._parser.add_argument('--distanceMeasure', metavar='<class>', help='Fully qualified class name for a DistanceMeasure implementation')
        self._parser.add_argument('--ds-debugOutput', metavar='<path>', help='Path for debug output')

        # DBSCAN-specific arguments
        self._parser.add_argument('-e', '--eps', required=True, metavar='<eps>', type=float,
                                help='Distance within which points are considered close enough to be assigned to one cluster')
        self._parser.add_argument('--npp', metavar='<npp>', type=int, default=DbscanSettings.defaultNumberOfPoints,
                                help=f'Number of points in partition (default: {DbscanSettings.defaultNumberOfPoints})')
        self._parser.add_argument('--numPts', required=True, metavar='<minPts>', type=int,
                                help='Minimum number of points to form a cluster (minPts)')
        self._parser.add_argument('--borderPointsAsNoise', action='store_true',
                                help='A flag indicating whether border points should be treated as noise')

        # DBSCAN++-specific arguments
        self._parser.add_argument('--samplingFraction', type=float, default=0.5,
                                  help='Fraction of points to sample for core points (default: 0.5)')
        self._parser.add_argument('--samplingStrategy', type=str, choices=['linspace', 'uniform', 'kcenters', 'kmeanspp'], default='linspace',
                                  help='Sampling strategy for core points (default: linspace)')

    def parse(self):
        namespace = self._parser.parse_args()

        # Common arguments
        self.args.masterUrl = namespace.ds_master
        self.args.pyFiles = namespace.ds_pyFiles
        self.args.inputPath = namespace.ds_input
        self.args.outputPath = namespace.ds_output

        # Parse distanceMeasure if provided
        if namespace.distanceMeasure:
            cls = globals()[namespace.distanceMeasure]
            self.args.distanceMeasure = cls()

        # Optional debug output
        if namespace.ds_debugOutput:
            self.args.debugOutputPath = namespace.ds_debugOutput

        # DBSCAN-specific arguments
        self.args.eps = namespace.eps
        self.args.numberOfPoints = namespace.npp
        self.args.minPts = namespace.numPts
        self.args.borderPointsAsNoise = namespace.borderPointsAsNoise

        # DBSCAN++-specific arguments
        self.args.samplingFraction = namespace.samplingFraction
        self.args.samplingStrategy = namespace.samplingStrategy

        return namespace


class IOHelper:
    # Static
    separator = ","

    @classmethod
    def readDataset(cls, sc: SparkContext, path: str, minPartitions: Optional[int] = None) -> RawDataSet:
        rawData = sc.textFile(path, minPartitions=minPartitions)
        return rawData.map(lambda line: Point(tuple(map(float, line.split(cls.separator)))))

    @classmethod
    def saveClusteringResult(cls, model: DbscanModel, outputPath: str) -> None:
        model.allPoints.map(
            lambda pt: ','.join((*map(str, pt.coordinates), str(pt.clusterId)))
        ).saveAsTextFile(outputPath)


class Dbscan(ABC):
    settings: DbscanSettings
    partitioningSettings: PartitioningSettings
    distanceAnalyzer: DistanceAnalyzer

    def __init__(
        self,
        settings: DbscanSettings,
        partitioningSettings: Optional[PartitioningSettings] = None
    ) -> None:
        super().__init__()
        self.settings = settings
        self.partitioningSettings = PartitioningSettings() if partitioningSettings is None else partitioningSettings
        self.distanceAnalyzer = DistanceAnalyzer(settings)

    @abstractmethod
    def run(self, data: RawDataSet) -> DbscanModel:
        raise NotImplementedError


class PartiallyMutablePoint(Point):
    tempId: TempPointId
    transientClusterId: ClusterId
    visited: bool

    def __init__(self, p: Point, tempId: TempPointId) -> None:
        super().__init__(p.coordinates, p.pointId, p.boxId, p.distanceFromOrigin, p.precomputedNumberOfNeighbors, p.clusterId, p.isSampled)
        self.tempId = tempId
        self.transientClusterId = p.clusterId
        self.visited = False

    @property
    def toImmutablePoint(self) -> Point:
        return Point(
            self.coordinates, self.pointId, self.boxId, self.distanceFromOrigin,
            self.precomputedNumberOfNeighbors, self.transientClusterId
        )


class DistributedDbscan(Dbscan, DistanceCalculation):
    distanceMeasure: DistanceMeasure

    def __init__(self,
        settings: DbscanSettings,
        partitioningSettings: Optional[PartitioningSettings] = None
    ) -> None:
        super().__init__(settings, partitioningSettings)
        self.distanceMeasure = settings.distanceMeasure

    def run(self, data: RawDataSet) -> DbscanModel:
        distanceAnalyzer = DistanceAnalyzer(self.settings)
        partitionedData = PointsPartitionedByBoxesRDD.apply(data, self.partitioningSettings, self.settings)

        DebugHelper.doAndSaveResult(
            data.context,
            "boxes",
            lambda path: data.context.parallelize(
                tuple(
                    ",".join(
                        map(
                            str,
                            (
                                box.bounds[0].lower,
                                box.bounds[1].lower,
                                box.bounds[0].upper,
                                box.bounds[1].upper,
                            ),
                        )
                    )
                    for box in partitionedData.boxes
                )
            ).saveAsTextFile(path),
        )

        partitionedData.rdd = self._sampleCorePointsPerPartition(partitionedData.rdd)

        pointsWithNeighborCounts = distanceAnalyzer.countNeighborsForEachPoint(partitionedData)
        broadcastBoxes = data.context.broadcast(partitionedData.boxes)

        distanceMeasure = self.distanceMeasure
        settings = self.settings
        partitioningSettings = self.partitioningSettings
        f_findClustersInOnePartition = self._findClustersInOnePartition

        def f_mapPartitions(partitionIndex: int, it: Iterable[tuple[PointSortKey, Point]]) -> Iterator[tuple[PointSortKey, Point]]:
            boxes = broadcastBoxes.value
            partitionBoundingBox = next(box for box in boxes if box.partitionId == partitionIndex)
            return f_findClustersInOnePartition(it, partitionBoundingBox, distanceMeasure, settings, partitioningSettings)

        partiallyClusteredData = pointsWithNeighborCounts.mapPartitionsWithIndex(
            f_mapPartitions,
            preservesPartitioning=True
        )

        partiallyClusteredData.persist()

        DebugHelper.doAndSaveResult(
            partiallyClusteredData.context,
            "partiallyClustered",
            lambda path: partiallyClusteredData.map(
                lambda x: ",".join((*map(str, x[1].coordinates), str(x[1].clusterId)))
            ).saveAsTextFile(path),
        )

        completelyClusteredData = self._mergeClustersFromDifferentPartitions(
            partiallyClusteredData, partitionedData.boxes
        )

        partiallyClusteredData.unpersist()

        return DbscanModel(completelyClusteredData, self.settings)

    def _sampleCorePointsPerPartition(self, data: RDD[tuple[PointSortKey, Point]]) -> RDD[tuple[PointSortKey, Point]]:
        frac = self.settings.samplingFraction
        if frac >= 1.0:
            return data

        broadcastFrac = data.context.broadcast(frac)
        f_computeDistance = self.distanceMeasure.compute

        def samplePartition_Uniform(it: Iterable[tuple[PointSortKey, Point]]) -> Iterable[tuple[PointSortKey, Point]]:
            partitionData = it
            if not isinstance(partitionData, Sequence):
                partitionData = tuple(partitionData)
            size = len(partitionData)
            if size == 0:
                yield from ()

            sampleSize = min(max(math.ceil(size * broadcastFrac.value), 1), size)
            subsetIndices = np.sort(np.random.choice(np.arange(size), sampleSize, replace=False))

            for i, x in enumerate(partitionData):
                yield (x[0], x[1].withSampling(i in subsetIndices))

        def samplePartition_Linspace(it: Iterable[tuple[PointSortKey, Point]]) -> Iterable[tuple[PointSortKey, Point]]:
            partitionData = it
            if not isinstance(partitionData, Sequence):
                partitionData = tuple(partitionData)
            size = len(partitionData)
            if size == 0:
                yield from ()

            sampleSize = min(max(math.ceil(size * broadcastFrac.value), 1), size)
            subsetIndices = np.linspace(0, size - 1, sampleSize, dtype=int)

            for i, x in enumerate(partitionData):
                yield (x[0], x[1].withSampling(i in subsetIndices))

        def samplePartition_KCenters(it: Iterable[tuple[PointSortKey, Point]]) -> Iterable[tuple[PointSortKey, Point]]:
            partitionData = it
            if not isinstance(partitionData, Sequence):
                partitionData = tuple(partitionData)
            size = len(partitionData)
            if size == 0:
                yield from ()

            dataPoints = np.array([x[1].coordinates for x in partitionData])

            sampleSize = min(max(math.ceil(size * broadcastFrac.value), 1), size)
            subsetIndices = np.empty(sampleSize, dtype=np.int_)

            # Initialize the first center to index 0
            centerId: np.int_ = np.int_(0)
            subsetIndices[0] = centerId

            # Precompute squared norms of all points
            normsSq = np.einsum('ij,ij->i', dataPoints, dataPoints)

            # Compute squared distances from all points to the first center
            closestDistSq = normsSq + normsSq[centerId] - 2 * np.dot(dataPoints, dataPoints[centerId])

            for c in range(1, sampleSize):
                # Select the point that is farthest from its closest center
                centerId = np.argmax(closestDistSq)
                subsetIndices[c] = centerId

                # Compute squared distances from all points to the new center
                distSqNewCenter = normsSq + normsSq[centerId] - 2 * np.dot(dataPoints, dataPoints[centerId])

                # Update closest distances
                np.minimum(closestDistSq, distSqNewCenter, out=closestDistSq)

            for i, x in enumerate(partitionData):
                yield (x[0], x[1].withSampling(i in subsetIndices))

        def samplePartition_KMeansPP(it: Iterable[tuple[PointSortKey, Point]]) -> Iterable[tuple[PointSortKey, Point]]:
            partitionData = it
            if not isinstance(partitionData, Sequence):
                partitionData = tuple(partitionData)
            size = len(partitionData)
            if size == 0:
                yield from ()

            dataPoints = np.array([x[1].coordinates for x in partitionData])

            sampleSize = min(max(math.ceil(size * broadcastFrac.value), 1), size)
            kmeans = KMeans(n_clusters=sampleSize, init="k-means++", random_state=42).fit(dataPoints)
            subsetIndices = set(np.argmin([f_computeDistance(point, center) for point in dataPoints]) for center in kmeans.cluster_centers_)

            for i, x in enumerate(partitionData):
                yield (x[0], x[1].withSampling(i in subsetIndices))

        if self.settings.samplingStrategy == 'uniform':
            return data.mapPartitions(samplePartition_Uniform, preservesPartitioning=True)

        elif self.settings.samplingStrategy == 'kcenters':
            return data.mapPartitions(samplePartition_KCenters, preservesPartitioning=True)

        elif self.settings.samplingStrategy == 'kmeanspp':
            return data.mapPartitions(samplePartition_KMeansPP, preservesPartitioning=True)

        else:  # self.settings.samplingStrategy == 'linspace'
            return data.mapPartitions(samplePartition_Linspace, preservesPartitioning=True)

    @classmethod
    def _findClustersInOnePartition(cls, it: Iterable[tuple[PointSortKey, Point]], boundingBox: Box, distanceMeasure: DistanceMeasure, settings: DbscanSettings, partitioningSettings: PartitioningSettings) -> Iterator[tuple[PointSortKey, Point]]:
        points = {1+tempPointId: PartiallyMutablePoint(x[1], 1+tempPointId) for tempPointId, x in enumerate(it)}

        partitionIndex = PartitionIndex(
            boundingBox, settings, partitioningSettings
        )
        partitionIndex.populate(points.values())

        startingPointWithId = cls._findUnvisitedCorePoint(points, settings)

        while startingPointWithId is not None:
            cls._expandCluster(points, partitionIndex, startingPointWithId[1], distanceMeasure, settings)
            startingPointWithId = cls._findUnvisitedCorePoint(points, settings)

        return ((PointSortKey(pt), pt.toImmutablePoint) for pt in points.values())

    @classmethod
    def _expandCluster(cls, points: dict[TempPointId, PartiallyMutablePoint], index: PartitionIndex, startingPoint: PartiallyMutablePoint, distanceMeasure: DistanceMeasure, settings: DbscanSettings) -> None:
        corePointsInCluster = {startingPoint.tempId}

        startingPoint.transientClusterId = startingPoint.pointId
        startingPoint.visited = True

        while corePointsInCluster:
            currentPointId = corePointsInCluster.pop()
            neighbors = cls._findUnvisitedNeighbors(index, points[currentPointId], distanceMeasure, settings)

            for n in neighbors:
                n.visited = True

                if n.isSampled and n.precomputedNumberOfNeighbors >= settings.numberOfPoints:
                    n.transientClusterId = startingPoint.transientClusterId
                    assert n.tempId != currentPointId
                    corePointsInCluster.add(n.tempId)
                elif not settings.treatBorderPointsAsNoise:
                    n.transientClusterId = startingPoint.transientClusterId
                else:
                    n.transientClusterId = DBSCAN_NOISE_POINT

    @classmethod
    def _findUnvisitedNeighbors(cls, index: PartitionIndex, pt: PartiallyMutablePoint, distanceMeasure: DistanceMeasure, settings: DbscanSettings) -> Iterator[PartiallyMutablePoint]:
        return (
            p
            for p in index.findClosePoints(pt)
            if (not p.visited and
                p.transientClusterId == DBSCAN_CLUSTER_UNDEF and
                cls.calculatePointDistance(p, pt, distanceMeasure) <= settings.epsilon)  # Potentially unnecessary since the distance is checked in index.findClosePoints
        )

    @staticmethod
    def _findUnvisitedCorePoint(points: dict[TempPointId, PartiallyMutablePoint], settings: DbscanSettings) -> Optional[tuple[TempPointId, PartiallyMutablePoint]]:
        return next((pt for pt in points.items() if not pt[1].visited and pt[1].isSampled and pt[1].precomputedNumberOfNeighbors >= settings.numberOfPoints), None)

    def _mergeClustersFromDifferentPartitions(self, partiallyClusteredData: RDD[tuple[PointSortKey, Point]], boxes: Iterable[Box]) -> RDD[Point]:
        distanceAnalyzer = DistanceAnalyzer(self.settings)
        pointsCloseToBoxBounds = distanceAnalyzer.findPointsCloseToBoxBounds(
            partiallyClusteredData, boxes, self.settings.epsilon
        ).filter(lambda pt: pt.isSampled)

        DebugHelper.doAndSaveResult(
            partiallyClusteredData.context,
            "cpdb",
            lambda path: pointsCloseToBoxBounds.map(
                lambda x: ",".join((*map(str, x.coordinates), str(x.clusterId)))
            ).saveAsTextFile(path),
        )

        mappings, borderPoints = self._generateMappings(pointsCloseToBoxBounds, boxes)

        broadcastMappings = partiallyClusteredData.context.broadcast(mappings)
        broadcastBorderPoints = partiallyClusteredData.context.broadcast(borderPoints)

        DebugHelper.doAndSaveResult(
            partiallyClusteredData.context,
            "mappings",
            lambda path: partiallyClusteredData.context.parallelize(tuple(map(str, mappings))).saveAsTextFile(path)
        )

        f_reassignClusterId = self._reassignClusterId

        def f_mapPartitions(it: Iterable[tuple[PointSortKey, Point]]) -> Iterator[Point]:
            m = broadcastMappings.value
            bp = broadcastBorderPoints.value
            return (f_reassignClusterId(x[1], m, bp) for x in it)

        return partiallyClusteredData.mapPartitions(f_mapPartitions)

    def _generateMappings(self, pointsCloseToBoxBounds: RDD[Point], boxes: Iterable[Box]) -> tuple[set[tuple[set[ClusterId], ClusterId]], dict[PointId, ClusterId]]:
        pointsInAdjacentBoxes = PointsInAdjacentBoxesRDD.apply(pointsCloseToBoxBounds, boxes).rdd

        settings = self.settings
        distanceMeasure = self.distanceMeasure
        f_calculatePointDistance = self.calculatePointDistance
        f_isCorePoint = self.isCorePoint
        f_addBorderPointToCluster_1 = self._addBorderPointToCluster_1

        def _mapParitions_pairs(it: Iterable[tuple[PairOfAdjacentBoxIds, Point]]) -> set[tuple[ClusterId, ClusterId]]:
            pointsInPartition = sorted((x[1] for x in it), key=lambda p: p.distanceFromOrigin)
            pairs: set[tuple[ClusterId, ClusterId]] = set()

            for i in range(1, len(pointsInPartition)):
                pi = pointsInPartition[i]
                assert pi.isSampled

                for j in range(i - 1, -1, -1):
                    pj = pointsInPartition[j]
                    assert pj.isSampled

                    if pi.distanceFromOrigin - pj.distanceFromOrigin > settings.epsilon:
                        break

                    if pi.boxId != pj.boxId and pi.clusterId != pj.clusterId and f_calculatePointDistance(pi, pj, distanceMeasure) <= settings.epsilon:
                        if settings.treatBorderPointsAsNoise:
                            enoughCorePoints = f_isCorePoint(pi, settings) and f_isCorePoint(pj, settings)
                        else:
                            enoughCorePoints = f_isCorePoint(pi, settings) or f_isCorePoint(pj, settings)

                        if enoughCorePoints:
                            c1, c2 = f_addBorderPointToCluster_1(pi, pj, settings)

                            if c1 != c2:
                                if pi.clusterId < pj.clusterId:
                                    pairs.add((pi.clusterId, pj.clusterId))
                                else:
                                    pairs.add((pj.clusterId, pi.clusterId))

            return pairs

        pairwiseMappings = pointsInAdjacentBoxes.mapPartitions(_mapParitions_pairs)

        borderPointsToBeAssignedToClusters: dict[PointId, ClusterId]
        if not settings.treatBorderPointsAsNoise:
            f_addBorderPointToCluster_2 = self._addBorderPointToCluster_2

            def _mapParitions_borderPoints(it: Iterable[tuple[PairOfAdjacentBoxIds, Point]]) -> Iterable[tuple[PointId, ClusterId]]:
                pointsInPartition = sorted((x[1] for x in it), key=lambda p: p.distanceFromOrigin)
                bp: dict[PointId, ClusterId] = {}

                for i in range(1, len(pointsInPartition)):
                    pi = pointsInPartition[i]
                    assert pi.isSampled

                    for j in range(i - 1, -1, -1):
                        pj = pointsInPartition[j]
                        assert pj.isSampled

                        if pi.distanceFromOrigin - pj.distanceFromOrigin > settings.epsilon:
                            break

                        if pi.boxId != pj.boxId and pi.clusterId != pj.clusterId and f_calculatePointDistance(pi, pj, distanceMeasure) <= settings.epsilon:
                            enoughCorePoints = f_isCorePoint(pi, settings) or f_isCorePoint(pj, settings)

                            if enoughCorePoints:
                                f_addBorderPointToCluster_2(pi, pj, settings, bp)

                return bp.items()

            borderPointsToBeAssignedToClusters = pointsInAdjacentBoxes.mapPartitions(_mapParitions_borderPoints).collectAsMap()
        else:
            borderPointsToBeAssignedToClusters = {}

        mappings: set[set[ClusterId]] = set()
        processedPairs: set[tuple[ClusterId, ClusterId]] = set()

        temp = pairwiseMappings.collect()

        for x in temp:
            self._processPairOfClosePoints(x[0], x[1], processedPairs, mappings)

        finalMappings = set((x, next(iter(x))) for x in mappings if len(x) > 0)

        return (finalMappings, borderPointsToBeAssignedToClusters)

    @classmethod
    def _addBorderPointToCluster_1(
        cls,
        pt1: Point,
        pt2: Point,
        settings: DbscanSettings
    ) -> tuple[ClusterId, ClusterId]:
        newClusterId1 = pt1.clusterId
        newClusterId2 = pt2.clusterId

        if  not settings.treatBorderPointsAsNoise:
            if not cls.isCorePoint(pt1, settings) and pt1.clusterId == DBSCAN_CLUSTER_UNDEF:
                newClusterId1 = pt2.clusterId
            elif not cls.isCorePoint(pt2, settings) and pt2.clusterId == DBSCAN_CLUSTER_UNDEF:
                newClusterId2 = pt1.clusterId

        return newClusterId1, newClusterId2

    @staticmethod
    def _processPairOfClosePoints(
        c1: ClusterId,
        c2: ClusterId,
        processedPairs: set[tuple[ClusterId, ClusterId]],
        mappings: set[set[ClusterId]]
    ) -> None:
        pair = (c1, c2) if c1 < c2 else (c2, c1)
        if pair not in processedPairs:
            processedPairs.add(pair)
            m1 = next((m for m in mappings if c1 in m), None)
            m2 = next((m for m in mappings if c2 in m), None)

            if m1 is None and m2 is None:
                mappings.add({c1, c2})
            elif m1 is None and m2 is not None:
                m2.add(c1)
            elif m1 is not None and m2 is None:
                m1.add(c2)
            elif m1 is not None and m2 is not None:
                if m1 != m2:
                    mappings.add(m1.union(m2))
                    m1.clear()
                    m2.clear()

    @classmethod
    def _addBorderPointToCluster_2(
        cls,
        pt1: Point,
        pt2: Point,
        settings: DbscanSettings,
        borderPointsToBeAssignedToClusters: dict[PointId, ClusterId]
    ) -> tuple[ClusterId, ClusterId]:
        newClusterId1 = pt1.clusterId
        newClusterId2 = pt2.clusterId

        if not settings.treatBorderPointsAsNoise:
            if not cls.isCorePoint(pt1, settings) and pt1.clusterId == DBSCAN_CLUSTER_UNDEF:
                borderPointsToBeAssignedToClusters[pt1.pointId] = pt2.clusterId
                newClusterId1 = pt2.clusterId
            elif not cls.isCorePoint(pt2, settings) and pt2.clusterId == DBSCAN_CLUSTER_UNDEF:
                borderPointsToBeAssignedToClusters[pt2.pointId] = pt1.clusterId
                newClusterId2 = pt1.clusterId

        return newClusterId1, newClusterId2

    @staticmethod
    def _reassignClusterId(
        pt: Point,
        mappings: set[tuple[set[ClusterId], ClusterId]],
        borderPoints: dict[PointId, ClusterId]
    ) -> Point:
        newClusterId = pt.clusterId

        if pt.pointId in borderPoints:
            newClusterId = borderPoints[pt.pointId]

        mapping = next((m for m in mappings if newClusterId in m[0]), None)

        if mapping is not None:
            newClusterId = mapping[1]

        if newClusterId == DBSCAN_CLUSTER_UNDEF:
            newClusterId = DBSCAN_NOISE_POINT

        return pt.withClusterId(newClusterId)

    @staticmethod
    def isCorePoint(pt: Point, settings: DbscanSettings) -> bool:
        return pt.isSampled and pt.precomputedNumberOfNeighbors >= settings.numberOfPoints

    @classmethod
    def train(
        cls,
        data: RawDataSet,
        settings: DbscanSettings,
        partitioningSettings: Optional[PartitioningSettings] = None
    ) -> DbscanModel:
        return cls(settings, PartitioningSettings() if partitioningSettings is None else partitioningSettings).run(data)
