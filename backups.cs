using System.Reflection.Emit;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using static Microsoft.ML.DataOperationsCatalog;
using System.Diagnostics;

string dataPath = Path.Combine(
    AppContext.BaseDirectory,
    "Data",
    "train_prepped.csv"
);

string docPath = Path.Combine(
    FindProjectDir(),
    "Metrics"
);
Directory.CreateDirectory(docPath);
Console.WriteLine(docPath);


MLContext mlContext = new MLContext(seed: 42);

TrainTestData splitDataView = LoadData(mlContext);


var devset = splitDataView.TrainSet;

// Not to be used before the end.
var testset = splitDataView.TestSet;

//var sample = mlContext.Data.CreateEnumerable<PitstopData>(devset, false).Take(5).ToList();
//foreach (var r in sample)
//{
  //  Console.WriteLine($"[{r.RaceYear}] {r.Race} {r.PitNextLap}");
    //var allKeys = mlContext.Data.CreateEnumerable<PitstopData>(devset, false).Select(r => r.RaceYear).Distinct().Count();
    //Console.WriteLine($"Rows: {mlContext.Data.CreateEnumerable<PitstopData>(devset, false).Count()}, distinct RaceYear: {allKeys}");
//}

var variants = new (string Name, bool useRace, bool useYear) []
{
    ("Base", false, false),
    ("Base+Year", false, true),
    ("Base+Race", true, false),
    ("Fase+Race+Year", true, true)
};

var groupings = new[]
{
    ("RaceYear", MakeGroupedFolds(mlContext, devset, 5, r => r.RaceYear)),
    ("Race", MakeGroupedFolds(mlContext, devset, 5, r => r.Race))
};

//IEstimator<ITransformer> preprocessing = BuildPreprocessor(mlContext);

var trainers = new Dictionary<string, IEstimator<ITransformer>>
{
    ["FastTree"] = mlContext.BinaryClassification.Trainers.FastTree(
        labelColumnName: "Label",
        featureColumnName: "Features"
    ),
    ["LightGBM"] = mlContext.BinaryClassification.Trainers.LightGbm(
        labelColumnName: "Label",
        featureColumnName: "Features"
    ),
    ["FastForest"] = mlContext.BinaryClassification.Trainers.FastForest(
        labelColumnName: "Label",
        featureColumnName: "Features"
    ),
    ["LogisticRegression"] = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
        labelColumnName: "Label",
        featureColumnName: "Features"
    )
};
// from earlier run
//var foldsByRaceYear = MakeGroupedFolds(mlContext, devset, 5, r => r.RaceYear);
//var foldsByRace = MakeGroupedFolds(mlContext, devset, 5, r => r.Race);

foreach (var (groupingName, folds) in groupings)
{

    foreach (var v in variants) 
    {
        if (groupingName == "Race" && v.useRace) continue;

        var preprocessing = BuildPreprocessor(mlContext, v.useRace, v.useYear);

        foreach (var (name, trainer) in trainers) {
            var stopWatch = System.Diagnostics.Stopwatch.StartNew();

            var cvResults = ManualCrossVal(
                mlContext, 
                folds,
                preprocessing,
                trainer
            );
            stopWatch.Stop();
            PrintValMetrics($"{name} | {v.Name} | grouped by ({groupingName})", cvResults, stopWatch.Elapsed.TotalSeconds);

        }
    }
}
//var results = ManualCrossVal(mlContext, devset, preprocessing, 5);
//PrintValMetrics("FastForest", results);


//ITransformer model = BuildAndTrainModel(mlContext, devset);

List<BinaryClassificationMetrics> ManualCrossVal(MLContext mlContext, List<(IDataView Train, IDataView Test)> folds, 
    IEstimator<ITransformer> preprocessor, IEstimator<ITransformer> trainer)
{
    var results = new List<BinaryClassificationMetrics>();
    var pipeline = preprocessor.AppendCacheCheckpoint(mlContext).Append(trainer);

    foreach (var fold in folds)
    {
        
        var model = pipeline.Fit(fold.Train);
        var predictions = model.Transform(fold.Test);

        var metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(
            predictions,
            labelColumnName: "Label",
            scoreColumnName: "Score"
        );
        results.Add(metrics);
    }
    return results;
}
List<(IDataView Train, IDataView Test)> MakeGroupedFolds(MLContext mlContext, IDataView data, int k, Func<PitstopData, string> key)
{
    var rows = mlContext.Data.CreateEnumerable<PitstopData>(data, reuseRowObject: false).ToList();

    // Assign whole race-years to folds: biggest first, each to the currently smallest fold
    var foldOf = new Dictionary<string, int>();
    var foldSize = new int[k];
    foreach (var g in rows.GroupBy(key)
                            .OrderByDescending(g => g.Count())
                            .ThenBy(g => g.Key))
    {
        int f = Array.IndexOf(foldSize, foldSize.Min());
        foldOf[g.Key] = f;
        foldSize[f] += g.Count();
    }
    var folds = new List<(IDataView, IDataView)>();
    for (int f = 0; f<k; f++)
    {
        var train = mlContext.Data.LoadFromEnumerable(rows.Where(r => foldOf[key(r)] != f).ToList());
        var test = mlContext.Data.LoadFromEnumerable(rows.Where(r => foldOf[key(r)] == f).ToList());
        folds.Add((train, test));
    }
    Console.WriteLine("Fold sizes: " + string.Join(", ", foldSize));
    return folds; 
}

void PrintValMetrics(string modelName, IEnumerable<BinaryClassificationMetrics> metrics, double time)
{
    var results = metrics.ToList();
    Console.WriteLine($"\nModel:   {modelName}");
    Console.WriteLine($"ROC AUC:   {results.Average(m => m.AreaUnderRocCurve):F4} ± {StandardDeviation(results.Select(m => m.AreaUnderRocCurve)):F4}");
    Console.WriteLine($"PR AUC:    {results.Average(m => m.AreaUnderPrecisionRecallCurve):F4} ± {StandardDeviation(results.Select(m => m.AreaUnderPrecisionRecallCurve)):F4}");
    Console.WriteLine($"Accuracy:  {results.Average(m => m.Accuracy):F4} ± {StandardDeviation(results.Select(m => m.Accuracy)):F4}");
    Console.WriteLine($"F1 Score:  {results.Average(m => m.F1Score):F4} ± {StandardDeviation(results.Select(m => m.F1Score)):F4}");
    Console.WriteLine($"Precision: {results.Average(m => m.PositivePrecision):F4} ± {StandardDeviation(results.Select(m => m.PositivePrecision)):F4}");
    Console.WriteLine($"Recall:    {results.Average(m => m.PositiveRecall):F4} ± {StandardDeviation(results.Select(m => m.PositiveRecall)):F4}");
    Console.WriteLine($"Time:      {time:F0}");
    using (StreamWriter outputFile = new StreamWriter(Path.Combine(docPath, "Metrics.txt"), true))
    {
        outputFile.WriteLine($"\nModel:   {modelName}");
        outputFile.WriteLine($"ROC AUC:   {results.Average(m => m.AreaUnderRocCurve):F4} ± {StandardDeviation(results.Select(m => m.AreaUnderRocCurve)):F4}");
        outputFile.WriteLine($"PR AUC:    {results.Average(m => m.AreaUnderPrecisionRecallCurve):F4} ± {StandardDeviation(results.Select(m => m.AreaUnderPrecisionRecallCurve)):F4}");
        outputFile.WriteLine($"Accuracy:  {results.Average(m => m.Accuracy):F4} ± {StandardDeviation(results.Select(m => m.Accuracy)):F4}");
        outputFile.WriteLine($"F1 Score:  {results.Average(m => m.F1Score):F4} ± {StandardDeviation(results.Select(m => m.F1Score)):F4}");
        outputFile.WriteLine($"Precision: {results.Average(m => m.PositivePrecision):F4} ± {StandardDeviation(results.Select(m => m.PositivePrecision)):F4}");
        outputFile.WriteLine($"Recall:    {results.Average(m => m.PositiveRecall):F4} ± {StandardDeviation(results.Select(m => m.PositiveRecall)):F4}");
        outputFile.WriteLine($"Time:      {time:F0}");
    }
}



TrainTestData LoadData(MLContext mlContext)
{
    IDataView rawData = mlContext.Data.LoadFromTextFile<PitstopData>(dataPath, separatorChar: ',', hasHeader: true);
    var rows = mlContext.Data.CreateEnumerable<PitstopData>(rawData, reuseRowObject: false)
        .Select(r => { r.RaceYear = $"{r.Race}|{r.Year}"; return r;});
    IDataView dataView = mlContext.Data.LoadFromEnumerable(rows);
    TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2,
        samplingKeyColumnName: "RaceYear", seed: 42);
    return splitDataView;
}

//ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainset)
//{
    //var pipeline = BuildPreprocessor(mlContext);

    //return pipeline.Fit(trainset);
//}

double StandardDeviation(IEnumerable<double> xs)
{
    var l = xs.ToList(); var m = l.Average();
    return Math.Sqrt(l.Sum(x => (x-m) * (x-m)) / (l.Count-1));
}

IEstimator<ITransformer> BuildPreprocessor(MLContext mlContext, bool useRace, bool useYear)
{

    var features = new List<string> {"CompoundEncoded"};
    IEstimator<ITransformer> pipe = 
        mlContext.Transforms.Conversion.ConvertType("Label", "PitNextLap", DataKind.Boolean)
            .Append(mlContext.Transforms.Categorical.OneHotEncoding("CompoundEncoded", "Compound"));
    
    if (useRace)
    {
        pipe = pipe.Append(mlContext.Transforms.Categorical.OneHotEncoding("RaceEncoded", "Race"));
        features.Add("RaceEncoded");
    }
    if (useYear) features.Add("Year");

    features.AddRange(new []
    {
        "PitStop", "LapNumber", "Stint", "TyreLife", "Position", "LapTime",
        "LapTime_Delta", "Cumulative_Degradation", "RaceProgress", "Position_Change"});

    return pipe.Append(mlContext.Transforms.Concatenate("Features", features.ToArray()));
}

string FindProjectDir()
{
    var dir = new DirectoryInfo(AppContext.BaseDirectory);
    while (dir != null && !dir.GetFiles("*.csproj").Any())
        dir = dir.Parent;
    return dir?.FullName ?? Directory.GetCurrentDirectory();
}