// DateTimeOffset keeps the transaction's own local time, so .Hour is the hour the
// customer saw, not the server's (DateTime would convert "+03:00" to server time)
public record PredictRequest(DateTimeOffset Timestamp, Dictionary<string, float> Features);

public record PredictResponse(float FraudProbability, bool IsFraud, float Threshold, string ModelVersion);
