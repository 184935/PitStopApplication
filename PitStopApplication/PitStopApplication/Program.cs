namespace PitStopApplication
{
    using Microsoft.Extensions.ML;
    using PitStopApplication.ML;
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // Add ML model.
            var modelPath = Path.Combine(
                builder.Environment.ContentRootPath,
                "Models",
                "model.zip");

            builder.Services
                .AddPredictionEnginePool<PitStopInput, PitstopPrediction>()
                .FromFile(
                    modelName: "PitStopModel",
                    filePath: modelPath,
                    watchForChanges: false);

            // Add services to the container.
            builder.Services.AddRazorPages();

            var app = builder.Build();

            // Configure the HTTP request pipeline.
            if (!app.Environment.IsDevelopment())
            {
                app.UseExceptionHandler("/Error");
                // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
                app.UseHsts();
            }

            app.UseHttpsRedirection();

            app.UseRouting();

            app.UseAuthorization();

            app.MapStaticAssets();
            app.MapRazorPages()
               .WithStaticAssets();

            app.Run();
        }
    }
}
