using System.Text.Json;

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

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseHttpsRedirection();

app.MapGet("/health", () => new { status = "ok" });
app.MapGet("/model-info", (ModelSchema s) => s);

app.Run();
