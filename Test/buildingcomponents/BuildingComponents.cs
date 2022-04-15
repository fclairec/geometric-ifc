using System.Windows.Media.Media3D;

namespace buildingcomponents
{
    public class BuildingComponents
    {
        /*private readonly MainWindow _mainWindow;

        public BuildingComponents(MainWindow mainWindow)
        {
            _mainWindow = mainWindow;
        }*/

        public int Id { get; set; }
        public string Globalid { get; set; }
        public string Objectname { get; set; }
        public string Objecttype { get; set; }
        public string ElementType { get; set; }
        public int Tag { get; set; }
        //public DiffuseMaterial Material { get; set; }
        //public double Thickness { get; set; }
        //public int? Storey { get; set; }
        //public bool Loadbearing { get; set; }
        //public bool Formwork { get; set; }
        //public double? Area { get; set; }
        //public double? Weight { get; set; }
        //public double? Volume { get; set; }

        //public double Area()
        //{
        //    var mytriangles = _mainWindow.MyTriangles.Where(t => t.ComponentId == Id);
        //    return mytriangles.Sum(triangle => triangle.Area);
        //}
    }
}