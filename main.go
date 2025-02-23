package main

import (
	"math/rand"
	"slices"
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

type RandomForest struct {
	trees       []*RegressionTree
	minFeatures int
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

func (rt *RegressionTree) fit(x [][]float64, y []float64) {
	featuresData := make(FeaturesWithTargets, len(x))

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

		featuresData[i] = feature_with_targets
	}

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

func (rf *RandomForest) fit(x [][]float64, y []float64) {
	iter := 100
	rf.trees = make([]*RegressionTree, iter)

	size := len(y)

	for j := range iter {
		println(j)

		bootstrap_start := rand.Intn(size)
		bootstrap_end := bootstrap_start + rand.Intn(size-bootstrap_start)

		// Defines how many features to select
		featureSelection := rf.minFeatures + rand.Intn(len(x)-rf.minFeatures)
		features := make([]int, featureSelection)
		for i := range featureSelection {
			pick := -1
			for pick == -1 {
				try := rand.Intn(len(x))
				if !slices.Contains(features, try) {
					pick = try
				}
			}
			features[i] = pick
		}

		bootstrap_x := make([][]float64, featureSelection)
		for i := range features {
			feature_slice := x[i][bootstrap_start:bootstrap_end]
			bootstrap_x[i] = feature_slice
		}
		bootstrap_y := y[bootstrap_start:bootstrap_end]

		rt := RegressionTree{maxDepth: 10, minSampleSize: 2}
		rt.fit(bootstrap_x, bootstrap_y)

		rf.trees[j] = &rt
	}
}

func (rf *RandomForest) predict(x []float64) float64 {
	predicted := 0.0
	for i := range rf.trees {
		predicted += rf.trees[i].predict(x, nil)
	}
	return predicted / float64(len(rf.trees))
}

func generateRandomFeatures(features int) []float64 {
	x := make([]float64, features)
	for i := range features {
		x[i] = rand.Float64()
	}
	return x
}

func generateRandomData(length int, features int) ([][]float64, []float64) {
	x := make([][]float64, 0)

	for range features {
		tmp := make([]float64, length)
		for range length {
			tmp = append(tmp, rand.Float64())
		}
		x = append(x, tmp)
	}

	y := make([]float64, length)
	for range length {
		y = append(y, rand.Float64())
	}

	return x, y
}

func main() {
	x, y := generateRandomData(1000, 20)

	rf := RandomForest{minFeatures: 10}
	rf.fit(x, y)

	test := generateRandomFeatures(20)
	res := rf.predict(test)
}
