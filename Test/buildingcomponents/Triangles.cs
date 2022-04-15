using System.Windows.Media.Media3D;

namespace buildingcomponents
{
    public class Triangles
    {
        /*private readonly MainWindow _mainWindow;

        public Triangles(MainWindow mainWindow)
        {
            _mainWindow = mainWindow;
        }*/

        //public int Id { get; set; }
        public int ComponentId { get; set; }
        public Point3D Point1 { get; set; }
        public Point3D Point2 { get; set; }
        public Point3D Point3 { get; set; }
        public Point3D Normal { get; set; }
        public double? Area { get; set; }

        //public double Area()
        //{
        //    var p1 = _mainWindow.MyVertices.Find(vt => vt.Id == PointId1);
        //    var p2 = _mainWindow.MyVertices.Find(vt => vt.Id == PointId2);
        //    var p3 = _mainWindow.MyVertices.Find(vt => vt.Id == PointId3);

        //    var a =
        //        Math.Sqrt(Math.Pow((double)(p2.X - p1.X), 2) + Math.Pow((double)(p2.Y - p1.Y), 2) +
        //                  Math.Pow((double)(p2.Z - p1.Z), 2));
        //    var b =
        //        Math.Sqrt(Math.Pow((double)(p3.X - p2.X), 2) + Math.Pow((double)(p3.Y - p2.Y), 2) +
        //                  Math.Pow((double)(p3.Z - p2.Z), 2));
        //    var c =
        //        Math.Sqrt(Math.Pow((double)(p1.X - p3.X), 2) + Math.Pow((double)(p1.Y - p3.Y), 2) +
        //                  Math.Pow((double)(p1.Z - p3.Z), 2));
        //    // SEE HERON'S FORMULA
        //    var s = (a + b + c) / 2;
        //    var myarea = Math.Sqrt(s * (s - a) * (s - b) * (s - c));
        //    return myarea;
        //}
    }
}