using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Microsoft.Extensions.ML;
using PitStopApplication.ML;

namespace PitStopApplication.Pages
{
    public class IndexModel : PageModel
    {
        private readonly PredictionEnginePool<PitStopInput, PitstopPrediction> predictionPool;
        public IndexModel(PredictionEnginePool<PitStopInput, PitstopPrediction> predictionPool)
        {
            this.predictionPool = predictionPool;
        }
        [BindProperty]
        public PitStopInput Input { get; set; } = new();
        public PitstopPrediction? Result { get; set; }
        public bool? PredictedPitstop { get; set; }
        public void OnGet()
        {

        }
        public void OnPost()
        {
            if (Input.TotalLaps <= 0 ||
                Input.LapNumber <= 0 ||
                Input.LapNumber > Input.TotalLaps)
            {
                ModelState.AddModelError("",
                    "Lap number must be between 0 and total laps, and total laps must be positive.");
                return;
            }
            if (!ModelState.IsValid) return;

            Input.RaceProgress = Input.LapNumber / Input.TotalLaps;

            Result = predictionPool.Predict("PitStopModel", Input);

            float threshold = 0.30f;
            PredictedPitstop = Result.Probability >= threshold;
        }
    }
}
