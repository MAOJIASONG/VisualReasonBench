using BurrPuzzle.DataCore;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Globalization;

namespace Luban.Original
{
    public sealed class ActionSpec
    {
        public string Kind = "move";
        public int[] PieceIds = Array.Empty<int>();
        public int Axis;
        public double Distance;
        public int[] Vector;
        public int Angle;
        public double[] Pivot;
        public int RuleIndex = -1;
    }
    public sealed class ActionResult { public bool Success; public string Message; public int[] ChangedPieceIds = Array.Empty<int>(); }
    public sealed class PieceState
    {
        public int Id;
        public string X, Y, Z;
        public int[] Orientation;
        public int HalfRotateAxis;
        // Search-side zone estimate; the set freezes only after victory, like MainLogic. NOT a Unity success receipt.
        public bool Finished;
    }
    public sealed class EngineState { public PieceState[] Pieces; public int Steps; }

    /// <summary>Original game's discrete action rules; no shortcut to solved/target layouts.</summary>
    public sealed class OriginalEngine
    {
        public DataTools Data { get; private set; }
        public double[] CenterLogical { get; private set; }
        public int Steps { get; private set; }
        public int FinishedCount => _finished.Count(x => x);
        public bool Solved => FinishedCount >= _finished.Length - 1;
        private bool[] _finished;
        private readonly Dictionary<AnglePolyhedron,BigInteger[]> _localBoundsCache = new Dictionary<AnglePolyhedron,BigInteger[]>();
        private readonly Dictionary<string,BigFraction> _translationCache = new Dictionary<string,BigFraction>();
        public long TranslationCacheHits { get; private set; }
        public long TranslationCacheMisses { get; private set; }
        // Exact pair geometry is translation invariant. Bounded cache survives search restores.
        private BigFraction PairTranslation(Polyhedron moving,Polyhedron obstacle,Direction direction)
        {
            string key=$"{moving.Index}:{obstacle.Index}:{direction.X},{direction.Y},{direction.Z}:"+
                $"{(int)moving.XSwitchAxisType},{(int)moving.YSwitchAxisType},{(int)moving.ZSwitchAxisType},{(int)moving.HalfRotateAxis}:"+
                $"{(int)obstacle.XSwitchAxisType},{(int)obstacle.YSwitchAxisType},{(int)obstacle.ZSwitchAxisType},{(int)obstacle.HalfRotateAxis}:"+
                $"{Exact(moving.Position.X-obstacle.Position.X)},{Exact(moving.Position.Y-obstacle.Position.Y)},{Exact(moving.Position.Z-obstacle.Position.Z)}";
            if(_translationCache.TryGetValue(key,out var result)){TranslationCacheHits++;return result;}
            TranslationCacheMisses++;
            result=MaySweepOverlap(moving,obstacle,direction)?Polyhedron.MoveBeforeIntersection(moving,obstacle,direction):null;
            if(_translationCache.Count>=100000)_translationCache.Clear();
            _translationCache[key]=result; return result;
        }
        private BigInteger[] LocalBounds(AnglePolyhedron hull)
        {
            if(!_localBoundsCache.TryGetValue(hull,out var bounds))
            {
                bounds=new[]{hull.Points.Min(p=>p.X),hull.Points.Min(p=>p.Y),hull.Points.Min(p=>p.Z),hull.Points.Max(p=>p.X),hull.Points.Max(p=>p.Y),hull.Points.Max(p=>p.Z)};
                _localBoundsCache[hull]=bounds;
            }
            return bounds;
        }
        // Necessary-only swept AABB test: rejection is exact; surviving pairs still use original SAT.
        private bool MaySweepOverlap(Polyhedron moving,Polyhedron obstacle,Direction direction)
        {
            var a=LocalBounds(moving.AnglePolyhedron);var b=LocalBounds(obstacle.AnglePolyhedron);
            var pa=new[]{moving.Position.X,moving.Position.Y,moving.Position.Z};var pb=new[]{obstacle.Position.X,obstacle.Position.Y,obstacle.Position.Z};
            var velocity=new[]{direction.X,direction.Y,direction.Z};BigFraction enter=BigFraction.Zero,exit=null;
            for(int axis=0;axis<3;axis++)
            {
                var lo=pb[axis]+new BigFraction(b[axis])-pa[axis]-new BigFraction(a[axis+3]);
                var hi=pb[axis]+new BigFraction(b[axis+3])-pa[axis]-new BigFraction(a[axis]);
                if(velocity[axis].IsZero){if(lo.CompareTo(BigFraction.Zero)>=0||hi.CompareTo(BigFraction.Zero)<=0)return false;continue;}
                var v=new BigFraction(velocity[axis]);var t0=lo/v;var t1=hi/v;
                if(t0.CompareTo(t1)>0){var temp=t0;t0=t1;t1=temp;}
                if(t0.CompareTo(enter)>0)enter=t0;
                if(exit==null||t1.CompareTo(exit)<0)exit=t1;
                if(exit.CompareTo(enter)<=0)return false;
            }
            return true;
        }
        public static OriginalEngine LoadFromJson(string json)
        {
            var token=JToken.Parse(json); NormalizeIntegers(token);
            var data = token.ToObject<PuzzleData>();
            if (data.polyhedrons == null || data.polyhedrons.Count < 2) throw new ArgumentException("Need at least two pieces");
            if (data.polyhedronTypes == null || data.polyhedronTypes.Count != data.polyhedrons.Count) throw new ArgumentException("Piece type count mismatch");
            data.moveRotates = data.moveRotates ?? new List<MoveRotate>();
            var engine = new OriginalEngine { Data = new DataTools { PuzzleData = data }, _finished = new bool[data.polyhedrons.Count] };
            engine.Data.LoadDirections(); engine.Data.LoadPolyhedrons();
            while (!engine.Data.IsAlreadyLoaded()) Thread.Sleep(5);
            foreach (var p in engine.Data.PolyhedronList) p.SwitchPlay1();
            var pos = engine.PositionsLogical();
            engine.CenterLogical = new[] { pos.Average(v => v[0]), pos.Average(v => v[1]), pos.Average(v => v[2]) };
            engine.UpdateFinishZones();
            return engine;
        }
        private static void NormalizeIntegers(JToken token)
        {
            foreach(var child in token.Children().ToArray()) NormalizeIntegers(child);
            if(token.Type==JTokenType.Float) { var value=(double)token; if(value==Math.Truncate(value)&&value>=long.MinValue&&value<=long.MaxValue)token.Replace(new JValue((long)value)); }
        }
        public EngineState Capture() => new EngineState {
            Steps=Steps,
            Pieces=Data.PolyhedronList.Select((p,i)=>new PieceState { Id=i, X=Exact(p.Position.X), Y=Exact(p.Position.Y), Z=Exact(p.Position.Z), Orientation=new[]{(int)p.XSwitchAxisType,(int)p.YSwitchAxisType,(int)p.ZSwitchAxisType}, HalfRotateAxis=(int)p.HalfRotateAxis, Finished=_finished[i]}).ToArray()
        };
        public void Restore(EngineState state)
        {
            if (state.Pieces.Length != Data.PolyhedronList.Count) throw new ArgumentException("Wrong state size");
            foreach(var s in state.Pieces)
            {
                var p=Data.PolyhedronList[s.Id];
                p.Position=new Position(Parse(s.X),Parse(s.Y),Parse(s.Z)); p.LastPosition=new Position(p.Position);
                p.XSwitchAxisType=(SwitchAxisType)s.Orientation[0]; p.YSwitchAxisType=(SwitchAxisType)s.Orientation[1]; p.ZSwitchAxisType=(SwitchAxisType)s.Orientation[2];
                p.HalfRotateAxis=(RotateAxis)s.HalfRotateAxis; p.AnglePolyhedron=p.AdjustAnglePolyhedron();
                p.Quaternion=VisualQuaternion(s.Id); _finished[s.Id]=s.Finished;
            }
            Steps=state.Steps;
        }
        public string StateKey() => string.Join(";", Capture().Pieces.Select(s=>$"{s.X},{s.Y},{s.Z}:{string.Join(",",s.Orientation)}:{s.HalfRotateAxis}:{(s.Finished?1:0)}"));
        public double[][] PositionsLogical() => Data.PolyhedronList.Select(p=>new[]{AsDouble(p.Position.X)/10000,AsDouble(p.Position.Y)/10000,AsDouble(p.Position.Z)/10000}).ToArray();
        public double[][] BoundsLogical() => Data.PolyhedronList.Select(p=> {
            var pts=p.AnglePolyhedron.Points; var pos=new[]{AsDouble(p.Position.X),AsDouble(p.Position.Y),AsDouble(p.Position.Z)};
            return new[]{(pos[0]+(double)pts.Min(q=>q.X))/10000,(pos[1]+(double)pts.Min(q=>q.Y))/10000,(pos[2]+(double)pts.Min(q=>q.Z))/10000,(pos[0]+(double)pts.Max(q=>q.X))/10000,(pos[1]+(double)pts.Max(q=>q.Y))/10000,(pos[2]+(double)pts.Max(q=>q.Z))/10000};
        }).ToArray();
        public UnityEngine.Quaternion VisualQuaternion(int id)
        {
            var p=Data.PolyhedronList[id]; int index=Array.FindIndex(Polyhedron.RotateAxisTypes,a=>a[0]==p.XSwitchAxisType&&a[1]==p.YSwitchAxisType&&a[2]==p.ZSwitchAxisType);
            var q=Polyhedron.RotateQuaternions[index];
            if(p.HalfRotateAxis != RotateAxis.None)
            {
                int r=(int)p.HalfRotateAxis-1; var axis=r/2==0?UnityEngine.Vector3.right:r/2==1?UnityEngine.Vector3.up:UnityEngine.Vector3.forward;
                q=UnityEngine.Quaternion.AngleAxis(r%2==0?45:-45,axis)*q;
            }
            return q;
        }
        public double MaxTranslation(int[] ids,int axis,int sign,double cap=1000)
        {
            ValidateAxis(axis); if(sign!=1&&sign!=-1)throw new ArgumentException("sign");
            var vector=new int[3];vector[axis]=sign;return MaxTranslationVector(ids,vector,cap);
        }
        public double MaxTranslationVector(int[] ids,int[] vector,double cap=1000)
        {
            ValidateIds(ids); var dir=VectorDirection(vector);
            if(!Finite(cap)||cap<0)throw new ArgumentException("cap");
            var best=Scaled(cap); var set=new HashSet<int>(ids);
            foreach(int id in ids)foreach(var other in Data.PolyhedronList)
            {
                if(set.Contains(other.Index))continue;
                var max=PairTranslation(Data.PolyhedronList[id],other,dir);
                if(max!=null&&max.CompareTo(best)<0)best=max;
            }
            return Math.Max(0,AsDouble(best)/10000);
        }
        public int[] BlockingClosure(int seed,int axis,int sign)
        {
            ValidateAxis(axis);if(sign!=1&&sign!=-1)throw new ArgumentException("sign");
            var vector=new int[3];vector[axis]=sign;return BlockingClosureVector(seed,vector);
        }
        public int[] BlockingClosureVector(int seed,int[] vector)
        {
            ValidateIds(new[]{seed});var dir=VectorDirection(vector);
            var set=new HashSet<int>{seed};var queue=new Queue<int>();queue.Enqueue(seed);
            while(queue.Count>0)
            {
                int id=queue.Dequeue();
                foreach(var other in Data.PolyhedronList)
                {
                    if(set.Contains(other.Index))continue;
                    var max=PairTranslation(Data.PolyhedronList[id],other,dir);
                    if(max!=null&&max.CompareTo(BigFraction.Zero)==0){set.Add(other.Index);queue.Enqueue(other.Index);}
                }
            }
            return set.OrderBy(i=>i).ToArray();
        }
        public int[] ApplicableSpecialRules() => Enumerable.Range(0,Data.PuzzleData.moveRotates.Count).Where(CheckSpecial).ToArray();
        public ActionResult Apply(ActionSpec action)
        {
            var before=Capture();
            try
            {
                List<Polyhedron> changed;
                switch(action.Kind)
                {
                    case "move":
                        ValidateIds(action.PieceIds);ValidateAxis(action.Axis);
                        if(!Finite(action.Distance)||Math.Abs(action.Distance)<1e-10)return Failure("Zero/nonfinite distance");
                        var dist=Scaled(Math.Abs(action.Distance));var sign=Math.Sign(action.Distance);
                        if(MaxTranslation(action.PieceIds,action.Axis,sign,Math.Abs(action.Distance))+1e-8<Math.Abs(action.Distance))return Failure("Translation blocked");
                        changed=Data.Move1SBeforeCross(action.PieceIds.ToList(),action.Axis*2+(sign<0?1:0),dist,false);
                        break;
                    case "move_vector":
                        ValidateIds(action.PieceIds);var vectorDir=VectorDirection(action.Vector);
                        if(!Finite(action.Distance)||action.Distance<=0)return Failure("Vector move needs positive finite scalar distance");
                        if(MaxTranslationVector(action.PieceIds,action.Vector,action.Distance)+1e-8<action.Distance)return Failure("Vector translation blocked");
                        changed=Data.Move1SBeforeCrossVector(action.PieceIds.ToList(),vectorDir,Scaled(action.Distance),false);
                        break;
                    case "rotate":
                        ValidateIds(action.PieceIds);ValidateAxis(action.Axis);
                        if(action.Angle!=45&&action.Angle!=-45&&action.Angle!=90&&action.Angle!=-90)return Failure("Use explicit ±45 or ±90 degree steps");
                        if(Math.Abs(action.Angle)==45&&Data.PuzzleData.disableHalfRotate)return Failure("Half rotations disabled for this original level");
                        double[] pivot=action.Pivot??PositionsLogical()[action.PieceIds[0]];
                        if(pivot.Length!=3||pivot.Any(v=>!Finite(v)))return Failure("Invalid pivot");
                        var center=new Position(Scaled(pivot[0]),Scaled(pivot[1]),Scaled(pivot[2]));
                        var rot=(RotateAxis)(1+action.Axis*2+(action.Angle<0?1:0));
                        changed=Data.RotateXYZ(action.PieceIds.ToList(),center,rot,Math.Abs(action.Angle)==45);
                        break;
                    case "special":
                        if(!CheckSpecial(action.RuleIndex))return Failure("Original special-rule preconditions not met");
                        changed=new List<Polyhedron>{Data.DoMoveRotate(Data.PuzzleData.moveRotates[action.RuleIndex])};
                        break;
                    default:return Failure("Unknown action kind");
                }
                if(changed.Count==0){Restore(before);return Failure("Action blocked");}
                foreach(var p in changed)p.Quaternion=VisualQuaternion(p.Index);
                Steps++;UpdateFinishZones();
                return new ActionResult{Success=true,Message="Accepted by restored original rules",ChangedPieceIds=changed.Select(p=>p.Index).ToArray()};
            }
            catch(Exception e){Restore(before);return Failure(e.GetType().Name+": "+e.Message);}
        }
        private bool CheckSpecial(int index)
        {
            if(index<0||index>=Data.PuzzleData.moveRotates.Count)return false;
            var r=Data.PuzzleData.moveRotates[index];
            return Data.CheckPolyhedronsConditionType(Data.PuzzleData.polyhedronTypes[Data.PuzzleData.polyhedronBaseIndex],
                Data.PolyhedronList.Select(p=>p.Position).ToList(),Data.PolyhedronList.Select(p=>new SwitchAxisTypeXYZ{x=p.XSwitchAxisType,y=p.YSwitchAxisType,z=p.ZSwitchAxisType}).ToList(),Data.PolyhedronList.Select(p=>p.HalfRotateAxis).ToList(),
                r.polyhedronConditions.Select(c=>c.positionScaled).ToList(),r.polyhedronConditions.Select(c=>c.switchAxisTypeXYZ).ToList(),r.polyhedronConditions.Select(c=>c.halfRotateAxis).ToList(),r.polyhedronConditions,r.polyhedronIndex);
        }
        private void UpdateFinishZones()
        {
            // Native MainLogic removes pieces leaving zones until victory latches.
            if(FinishedCount>=_finished.Length-1)return;
            Array.Clear(_finished,0,_finished.Length);
            var bounds=BoundsLogical();
            for(int i=0;i<bounds.Length;i++)for(int axis=0;axis<3;axis++)foreach(int sign in new[]{-1,1})
            {
                bool overlap=true;
                for(int a=0;a<3;a++)
                {
                    double lo=a==axis?(sign>0?150:-450):-300,hi=a==axis?(sign>0?450:-150):300;
                    if(bounds[i][a]-CenterLogical[a]>=hi||bounds[i][a+3]-CenterLogical[a]<=lo){overlap=false;break;}
                }
                if(overlap)_finished[i]=true;
            }
        }
        private void ValidateIds(int[] ids){if(ids==null||ids.Length==0||ids.Distinct().Count()!=ids.Length||ids.Any(i=>i<0||i>=Data.PolyhedronList.Count))throw new ArgumentException("Invalid piece IDs");}
        private static void ValidateAxis(int axis){if(axis<0||axis>2)throw new ArgumentException("Invalid axis");}
        private static Direction VectorDirection(int[] vector)
        {
            if(vector==null||vector.Length!=3)throw new ArgumentException("Vector needs three integer components");
            var gcd=BigInteger.GreatestCommonDivisor(BigInteger.GreatestCommonDivisor(vector[0],vector[1]),vector[2]);
            if(gcd!=BigInteger.One)throw new ArgumentException("Vector must be nonzero and primitive (component GCD=1)");
            return new Direction(vector[0],vector[1],vector[2]);
        }
        private static Direction DirectionFor(int axis,int sign)=>new Direction(axis==0?sign:0,axis==1?sign:0,axis==2?sign:0);
        private static bool Finite(double x)=>!double.IsNaN(x)&&!double.IsInfinity(x);
        private static ActionResult Failure(string message)=>new ActionResult{Success=false,Message=message};
        public static double AsDouble(BigFraction f)=>(double)f.Numerator/(double)f.Denominator;
        public static string Exact(BigFraction f)=>f.Numerator.ToString(CultureInfo.InvariantCulture)+"/"+f.Denominator.ToString(CultureInfo.InvariantCulture);
        public static BigFraction Parse(string value){var p=value.Split('/');return new BigFraction(BigInteger.Parse(p[0],CultureInfo.InvariantCulture),p.Length==1?BigInteger.One:BigInteger.Parse(p[1],CultureInfo.InvariantCulture));}
        private static BigFraction Scaled(double logical)=>new BigFraction(new BigInteger(Math.Round(logical*100000000)),new BigInteger(10000));
    }
}
