using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;

public class ApiTests
{
    private readonly HttpClient client;

    public ApiTests()
    {
        // Tests run from api.Tests/bin/Debug/net10.0, so climb 4 folders to the repo root
        var schemaPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../models/model_schema.json"));
        Environment.SetEnvironmentVariable("Model__SchemaPath", schemaPath);

        client = new WebApplicationFactory<Program>().CreateClient();
    }

    [Fact]
    public async Task Health_returns_ok()
    {
        var response = await client.GetAsync("/health");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task Predict_flags_known_fraud()
    {
        var request = new
        {
            timestamp = "2026-10-10T20:15:00",
            features = new Dictionary<string, float>
            {
                // The fraud example from FraudApi.http (a real test-set fraud, P(fraud) ~ 0.9999)
                ["V1"] = -1.548788f, ["V2"] = 1.808698f, ["V3"] = -0.953509f, ["V4"] = 2.213085f,
                ["V5"] = -2.015728f, ["V6"] = -0.913457f, ["V7"] = -2.356013f, ["V8"] = 1.197169f,
                ["V9"] = -1.678374f, ["V10"] = -3.53865f, ["V11"] = 3.10209f, ["V12"] = -3.993373f,
                ["V13"] = -1.937411f, ["V14"] = -3.822894f, ["V15"] = 0.83097f, ["V16"] = -2.475359f,
                ["V17"] = -5.211875f, ["V18"] = -0.413872f, ["V19"] = 0.933262f, ["V20"] = 0.390786f,
                ["V21"] = 0.855138f, ["V22"] = 0.774745f, ["V23"] = 0.059037f, ["V24"] = 0.3432f,
                ["V25"] = -0.468938f, ["V26"] = -0.278338f, ["V27"] = 0.625922f, ["V28"] = 0.395573f,
                ["Amount"] = 76.94f,
            },
        };

        var response = await client.PostAsJsonAsync("/predict", request);
        var prediction = await response.Content.ReadFromJsonAsync<PredictResponse>();

        Assert.False(prediction!.IsFraud);
    }
}
