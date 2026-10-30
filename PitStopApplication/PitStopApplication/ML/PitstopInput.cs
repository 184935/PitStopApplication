namespace PitStopApplication.ML
{
    using Microsoft.ML.Data;
    public class PitStopInput
    {
        public string Compound { get; set; } = "MEDIUM";
        public float Year { get; set; }
        public float PitStop { get; set; }
        public float LapNumber { get; set; }
        public float Stint {  get; set; }
        public float TyreLife { get; set;  }
        public float Position { get; set; }
        public float RaceProgress { get; set; }
        public float Position_Change { get; set; }
        public float TotalLaps { get; set; }

        public float PitNextLap { get; set; }
    }
}
