using System.ComponentModel;
using Microsoft.ML.Data;

public class PitstopData
{
    [LoadColumn(1)]
    public string? Driver;

    [LoadColumn(2)]
    public string? Compound;

    [LoadColumn(3)]
    public string? Race;

    [LoadColumn(4)]
    public float Year;

    [LoadColumn(5)]
    public float PitStop;

    [LoadColumn(6)]
    public float LapNumber;

    [LoadColumn(7)]
    public float Stint;

    [LoadColumn(8)]
    public float TyreLife;

    [LoadColumn(9)]
    public float Position;

    [LoadColumn(10)]
    public float LapTime;

    [LoadColumn(11)]
    public float LapTime_Delta;

    [LoadColumn(12)]
    public float Cumulative_Degradation;

    [LoadColumn(13)]
    public float RaceProgress;

    [LoadColumn(14)]
    public float Position_Change;

    [LoadColumn(15)]
    public float PitNextLap; 
    [LoadColumn(16)]
    public string? RaceYear {get; set;} = "";
}

public class PitStopPrediction : PitstopData
{
    [ColumnName("PredictedLabel")]
    public bool Prediction {get; set;}

    public float Probability {get; set;}
    public float Score {get; set;}
}

public class ScoredRow
{
    public bool Label {get; set;}
    public float Probability {get; set;}

}