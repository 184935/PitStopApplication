namespace PitStopApplication.ML
{
    using Microsoft.ML.Data;
    public class PitstopPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedLabel { get; set; }
        public float Score { get; set; }
        public float Probability { get; set; }
    }
}
