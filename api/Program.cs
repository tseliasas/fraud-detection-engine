using System.Text.Json;
using Microsoft.ML.OnnxRuntime;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
builder.Services.AddOpenApi();

// --- Model contract ---
var schemaPath  = builder.Configuration["Model:SchemaPath"]!;
var schemaJson  = File.ReadAllText(schemaPath);
var jsonOptions = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };
var schema      = JsonSerializer.Deserialize<ModelSchema>(schemaJson, jsonOptions)!;

// Fail at startup instead of serving predictions from a half-loaded contract
if (schema.Features.Length == 0)
    throw new InvalidOperationException("Model schema has no features. Check model_schema.json.");
if (schema.Threshold <= 0 || schema.Threshold >= 1)
    throw new InvalidOperationException($"Model threshold {schema.Threshold} must be between 0 and 1.");

builder.Services.AddSingleton(schema);

// --- ONNX model ---
// The schema names the model file; it lives next to the schema
var modelDir  = Path.GetDirectoryName(schemaPath)!;
var modelPath = Path.Combine(modelDir, schema.ModelFile);
var session   = new InferenceSession(modelPath);

if (!session.InputMetadata.ContainsKey(schema.InputName))
    throw new InvalidOperationException($"ONNX model has no input named '{schema.InputName}'.");
if (!session.OutputMetadata.ContainsKey(FraudPredictor.ProbabilitiesOutput))
    throw new InvalidOperationException($"ONNX model has no output named '{FraudPredictor.ProbabilitiesOutput}'.");

// Loading the model is slow, so do it once and share it across requests
builder.Services.AddSingleton(session);
builder.Services.AddSingleton<FraudPredictor>();

// --- CORS ---
// Browsers block a page on one origin (the dashboard) from calling another (this API) unless allowed
var allowedOrigins = builder.Configuration.GetSection("Cors:AllowedOrigins").Get<string[]>() ?? [];
builder.Services.AddCors(options =>
    options.AddDefaultPolicy(policy =>
        policy.WithOrigins(allowedOrigins).AllowAnyHeader().WithMethods("GET", "POST")));

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseHttpsRedirection();
app.UseCors();

app.MapGet("/health", () => new { status = "ok" });
app.MapGet("/model-info", (ModelSchema s) => s);

app.MapPost("/predict", (PredictRequest request, FraudPredictor predictor) =>
{
    var missing = predictor.FindMissingFeatures(request);
    if (missing.Count > 0)
        return Results.BadRequest(new { error = "Missing features", missing });

    return Results.Ok(predictor.Predict(request));
});

app.Run();
