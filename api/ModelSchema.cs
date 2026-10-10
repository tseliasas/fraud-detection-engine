public class ModelSchema
{
    public string ModelFile { get; set; } = "";
    public string InputName { get; set; } = "";
    public string Version { get; set; } = "";
    public string[] Features { get; set; } = [];
    public float Threshold { get; set; }
}
