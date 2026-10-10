using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// Turns a transaction into the model's 30 inputs and scores it, following model_schema.json
public class FraudPredictor(ModelSchema schema, InferenceSession session)
{
    public const string ProbabilitiesOutput = "probabilities";
    const string HourFeature = "hour";

    // Every schema feature except "hour" must be sent by the caller; hour comes from the timestamp
    public List<string> FindMissingFeatures(PredictRequest request)
    {
        var sent = request.Features ?? [];

        return schema.Features
            .Where(name => name != HourFeature && !sent.ContainsKey(name))
            .ToList();
    }

    public PredictResponse Predict(PredictRequest request)
    {
        // The schema decides the order, not the caller
        var values = new float[schema.Features.Length];
        for (int i = 0; i < schema.Features.Length; i++)
        {
            var name = schema.Features[i];
            values[i] = name == HourFeature ? request.Timestamp.Hour : request.Features[name];
        }

        // One row of 30 floats, the same [None, 30] shape the model was exported with
        var tensor = new DenseTensor<float>(values, [1, values.Length]);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(schema.InputName, tensor) };

        using var results = session.Run(inputs);
        var probabilities = results.First(r => r.Name == ProbabilitiesOutput).AsTensor<float>();

        // Column 1 is P(fraud), same as predict_proba(...)[:, 1] in Python
        var fraudProbability = probabilities[0, 1];

        return new PredictResponse(
            fraudProbability,
            fraudProbability >= schema.Threshold,
            schema.Threshold,
            schema.Version);
    }
}
