package main

import (
	"fmt"
	"sort"
)

// Require sorted lists of the features and have to retrieve target matching this feature so each feature
// has its own set of targets after sorting
type (
	FeatureWithTarget   = [][2]float64
	FeaturesWithTargets = []FeatureWithTarget
)

type Node struct {
	left       *Node
	right      *Node
	idx        int
	splitValue float64
	prediction float64
}

type RegressionTree struct {
	root                 *Node
	minSampleSize        int
	maxDepth             int
	minVarianceReduction float64
}

func variance(arr []float64) float64 {
	sum := 0.0
	for i := range arr {
		sum += arr[i]
	}
	mean := sum / float64(len(arr))

	var_sum := 0.0
	for i := range arr {
		var_sum += arr[i] - mean
	}

	return var_sum * var_sum
}

func splitFeatures(featuresData FeaturesWithTargets, featureIdx int, splitValue float64) (FeaturesWithTargets, FeaturesWithTargets) {
	splitIdx := 0
	for i := range featuresData[featureIdx] {
		if featuresData[featureIdx][i][0] > splitValue {
			break
		} else {
			splitIdx++
		}
	}

	leftSplit := make(FeaturesWithTargets, 0)
	rightSplit := make(FeaturesWithTargets, 0)

	for i := range featuresData {
		featureLeftSplit := make(FeatureWithTarget, 0)
		featureRightSplit := make(FeatureWithTarget, 0)

		for j := range featuresData[i] {
			if j < splitIdx {
				featureLeftSplit = append(featureLeftSplit, featuresData[i][j])
			} else {
				featureRightSplit = append(featureRightSplit, featuresData[i][j])
			}
		}
		leftSplit = append(leftSplit, featureLeftSplit)
		rightSplit = append(rightSplit, featureRightSplit)
	}

	return leftSplit, rightSplit
}

func splitSingleFeatureTarget(featureData FeatureWithTarget, splitValue float64) ([]float64, []float64) {
	leftSplit := make([]float64, 0)
	rightSplit := make([]float64, 0)

	for i := range featureData {
		if featureData[i][0] > splitValue {
			leftSplit = append(leftSplit, featureData[i][1])
		} else {
			rightSplit = append(rightSplit, featureData[i][1])
		}
	}
	return leftSplit, rightSplit
}

func varianceReduction(featureData FeatureWithTarget, splitValue float64) float64 {
	targets := make([]float64, len(featureData))
	for i := range featureData {
		targets[i] = featureData[i][1]
	}
	parentVariance := variance(targets)

	leftSplit, rightSplit := splitSingleFeatureTarget(featureData, splitValue)
	leftSplitVariance := variance(leftSplit)
	rightSplitVariance := variance(rightSplit)

	weightedVariance := float64(len(leftSplit)/len(targets))*leftSplitVariance + float64(len(rightSplit)/len(targets))*rightSplitVariance
	return parentVariance - weightedVariance
}

func findBestSplit(featuresData FeaturesWithTargets) (float64, int, float64) {
	bestVarianceReduction := 0.0
	// These are null values, they should be overwritten in all cases
	bestFeatureIndex := -1
	bestSplitValue := 0.0

	// Iterate over features
	for i := range featuresData {
		for j := range len(featuresData[i]) - 1 {
			splitValue := (featuresData[i][j][0] + featuresData[i][j+1][0]) / 2
			varianceReduction := varianceReduction(featuresData[i], splitValue)
			if varianceReduction > bestVarianceReduction {
				bestVarianceReduction = varianceReduction
				bestFeatureIndex = i
				bestSplitValue = splitValue
			}
		}
	}
	return bestVarianceReduction, bestFeatureIndex, bestSplitValue
}

func (rt *RegressionTree) buildTree(featuresData FeaturesWithTargets, depth int) *Node {
	samples := len(featuresData[0])

	prediction := 0.0
	for i := range featuresData {
		for j := range featuresData[i] {
			prediction += featuresData[i][j][1]
		}
	}
	prediction = prediction / float64(len(featuresData[0]))

	if samples < rt.minSampleSize {
		return &Node{prediction: prediction}
	}

	// Check stopping condition
	if depth > rt.maxDepth {
		return &Node{prediction: prediction}
	}

	varianceReduction, featureIndex, splitValue := findBestSplit(featuresData)
	if featureIndex == -1 || varianceReduction < rt.minVarianceReduction {
		return &Node{prediction: prediction}
	}

	leftSplit, rightSplit := splitFeatures(featuresData, featureIndex, splitValue)
	leftNode := rt.buildTree(leftSplit, depth+1)
	rightNode := rt.buildTree(rightSplit, depth+1)
	return &Node{left: leftNode, right: rightNode, idx: featureIndex, splitValue: splitValue, prediction: prediction}
}

func (rt *RegressionTree) fit(featuresData FeaturesWithTargets) {
	rt.root = rt.buildTree(featuresData, 0)
}

func (rt *RegressionTree) predict(x []float64, node *Node) float64 {
	if node == nil {
		node = rt.root
	}

	if node.left == nil && node.right == nil {
		return node.prediction
	}

	if x[node.idx] <= node.splitValue {
		return rt.predict(x, node.left)
	} else {
		return rt.predict(x, node.right)
	}
}

func prepareData(x [][]float64, y []float64) FeaturesWithTargets {
	features_with_targets := make(FeaturesWithTargets, len(x))

	// Iterate over features, joining them with target, sorting
	for i := range len(x) {

		feature_with_targets := make(FeatureWithTarget, len(y))
		// Iterate over targets and feature
		for j := range x[i] {
			joined := [2]float64{x[i][j], y[j]}
			feature_with_targets[j] = joined
		}

		sort.Slice(feature_with_targets, func(i, j int) bool {
			return feature_with_targets[i][0] < feature_with_targets[j][0]
		})

		features_with_targets[i] = feature_with_targets
	}

	return features_with_targets
}

func main() {
	x := [][]float64{
		{2.4, 1.2, 6.4, 7.3},
		{2.1, 4.7, 8.1, 6.2},
	}
	y := []float64{8.1, 13.0, 16.0, 27.0}

	features_with_targets := prepareData(x, y)

	rt := RegressionTree{maxDepth: 3, minSampleSize: 2}
	rt.fit(features_with_targets)

	test := rt.predict([]float64{2.4, 2.1}, nil)
	test1 := rt.predict([]float64{1.2, 4.7}, nil)
	test2 := rt.predict([]float64{6.4, 8.1}, nil)
	test3 := rt.predict([]float64{7.3, 6.2}, nil)
	fmt.Printf("%f\n", test)
	fmt.Printf("%f\n", test1)
	fmt.Printf("%f\n", test2)
	fmt.Printf("%f\n", test3)
}
