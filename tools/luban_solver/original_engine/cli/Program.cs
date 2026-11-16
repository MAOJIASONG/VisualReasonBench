using System.Diagnostics;
using System.Text.Json;
using Luban.Original;
if(args[0]=="--compare-native")
{
    var all=new List<object>();int tested=0,mismatches=0;
    using var probes=JsonDocument.Parse(File.ReadAllText(args[2]));
    foreach(var level in probes.RootElement.GetProperty("levels").EnumerateArray())
    {
        int id=level.GetProperty("level").GetInt32();
        var core=OriginalEngine.LoadFromJson(File.ReadAllText(Path.Combine(args[1],$"Puzzle_{id}.json")));
        int checks=0,failed=0;var rows=new List<object>();
        foreach(var action in level.GetProperty("actions").EnumerateArray())
        {
            if(action.GetProperty("command").GetString()!="move")continue;
            var p=action.GetProperty("params"); int piece=p.GetProperty("piece_id").GetInt32(),axis="xyz".IndexOf(p.GetProperty("axis").GetString()!);
            double dist=p.GetProperty("distance").GetDouble();bool exploratory=p.GetProperty("is_exploratory").GetBoolean();
            double limit=core.MaxTranslation(new[]{piece},axis,Math.Sign(dist),Math.Abs(dist));
            bool ours=exploratory?limit>1e-8:limit+1e-8>=Math.Abs(dist);
            bool native=action.GetProperty("status").GetString()=="success";
            checks++;tested++;if(ours!=native){failed++;mismatches++;}
            rows.Add(new{piece,axis,dist,exploratory,limit,ours,native,match=ours==native});
        }
        all.Add(new{level=id,checks,failed,rows});Console.Error.WriteLine($"level={id} checks={checks} mismatches={failed}");
    }
    File.WriteAllText(args[3],JsonSerializer.Serialize(new{tested,mismatches,levels=all},new JsonSerializerOptions{WriteIndented=true}));
    Console.WriteLine($"NATIVE_PARITY tested={tested} mismatches={mismatches}");return;
}
var timer=Stopwatch.StartNew();
var engine=OriginalEngine.LoadFromJson(File.ReadAllText(args[0]));
Console.WriteLine("loaded_seconds="+timer.Elapsed.TotalSeconds);
var before=engine.Capture();var key=engine.StateKey();
var specials=engine.ApplicableSpecialRules();
Console.WriteLine("specials="+string.Join(",",specials));
void Require(bool ok,string message){if(!ok)throw new Exception(message);}
if(args.Contains("--special4"))
{
    Require(specials.Contains(1),"Expected initial level4 special rule1");
    var result=engine.Apply(new ActionSpec{Kind="special",RuleIndex=1});Require(result.Success,result.Message);
    var piece=engine.Capture().Pieces[2];Require(piece.Y=="600000/1"&&piece.Orientation.SequenceEqual(new[]{2,1,4}),"Original rule1 expected pose mismatch");
    Require(engine.StateKey()!=key,"Special failed to change state");
    engine.Restore(before);Require(engine.StateKey()==key,"Restore is not exact");
    var blocked=engine.Apply(new ActionSpec{Kind="special",RuleIndex=-1});Require(!blocked.Success&&engine.StateKey()==key,"Invalid special mutated state");
    // With three pieces, one zone entry is not victory and leaving removes it.
    var grouped=engine.Apply(new ActionSpec{Kind="move",PieceIds=new[]{0,1,2},Axis=0,Distance=1000});Require(grouped.Success,grouped.Message);
    var displaced=engine.Capture();Require(engine.FinishedCount==0,"Beyond all finish slabs should not latch victory");
    engine.Restore(before);
    Console.WriteLine("SPECIAL_AND_RESTORE_PASS");
}
if(args.Contains("--self-test"))
{
    var initial=engine.Capture();var initialKey=engine.StateKey();
    Require(OriginalEngine.AsDouble(OriginalEngine.Parse("-400000/1"))==-400000,"Negative fraction conversion");
    Require(OriginalEngine.AsDouble(OriginalEngine.Parse("3/2"))==1.5,"Fraction conversion must not truncate");
    var blocked=engine.Apply(new ActionSpec{Kind="move",PieceIds=new[]{0},Axis=0,Distance=1});
    Require(!blocked.Success,"Expected blocked firstpiece x+1 inlevel0");Require(initialKey==engine.StateKey(),"Rejected action mutated state");
    Require(engine.BlockingClosure(0,0,1).SequenceEqual(new[]{0,1}),"Blocked closure omitted second piece");
    var group=engine.Apply(new ActionSpec{Kind="move",PieceIds=new[]{0,1},Axis=0,Distance=2});Require(group.Success,group.Message);Require(engine.PositionsLogical()[0][0]==-38,"Group requested distance incorrect");engine.Restore(initial);
    var move=engine.Apply(new ActionSpec{Kind="move",PieceIds=new[]{0},Axis=2,Distance=-300});Require(move.Success,move.Message);
    Require(engine.PositionsLogical()[0][2]==-360,"Requested distance not exact");
    var afterMove=engine.Capture();Require(engine.FinishedCount==1,"Expected finish zone");var movedKey=engine.StateKey();
    var half=engine.Apply(new ActionSpec{Kind="rotate",PieceIds=new[]{0},Axis=2,Angle=45});Require(half.Success,half.Message);
    Require(engine.Capture().Pieces[0].HalfRotateAxis!=0,"45 degree state missing");
    var halfState=engine.Capture();var halfKey=engine.StateKey();
    var second=engine.Apply(new ActionSpec{Kind="rotate",PieceIds=new[]{0},Axis=2,Angle=45});Require(second.Success,second.Message);var twoHalves=engine.StateKey();
    engine.Restore(afterMove);Require(engine.StateKey()==movedKey,"Restore aftermove mismatch");
    var whole=engine.Apply(new ActionSpec{Kind="rotate",PieceIds=new[]{0},Axis=2,Angle=90});Require(whole.Success,whole.Message);Require(engine.StateKey()==twoHalves,"Two45 steps != one90");
    engine.Restore(halfState);Require(engine.StateKey()==halfKey,"Half restore missing orientation");
    var back=engine.Apply(new ActionSpec{Kind="rotate",PieceIds=new[]{0},Axis=2,Angle=-45});Require(back.Success&&engine.StateKey()==movedKey,"Inverse half rotation mismatch");
    var diagonal=engine.Apply(new ActionSpec{Kind="move_vector",PieceIds=new[]{0},Vector=new[]{1,1,0},Distance=2});Require(diagonal.Success,diagonal.Message);Require(engine.PositionsLogical()[0][0]==-38&&engine.PositionsLogical()[0][1]==102,"Vector scalar displacement mismatch");
    engine.Restore(initial);Require(engine.StateKey()==initialKey,"Initial snapshot mutated");
    var noOp=engine.Apply(new ActionSpec{Kind="rotate",PieceIds=new[]{0},Axis=0,Angle=180});Require(!noOp.Success,"180 requires explicit steps");Require(engine.StateKey()==initialKey,"Rejected angle mutatedstate");
    Console.WriteLine("BLOCKED_GROUP_VECTOR_HALF_ROTATION_AND_RESTORE_PASS");
}
if(args.Contains("--profile"))
{
    var profile=Stopwatch.StartNew();var first=engine.MaxTranslation(new[]{0},0,1,600);double firstMs=profile.Elapsed.TotalMilliseconds;
    profile.Restart();var cached=engine.MaxTranslation(new[]{0},0,1,600);double cachedMs=profile.Elapsed.TotalMilliseconds;
    profile.Restart();var rotate=engine.Apply(new ActionSpec{Kind="rotate",PieceIds=new[]{0},Axis=0,Angle=45});double rotateMs=profile.Elapsed.TotalMilliseconds;
    Console.WriteLine(JsonSerializer.Serialize(new{first,cached,firstMs,cachedMs,rotateMs,rotate.Success,engine.TranslationCacheHits,engine.TranslationCacheMisses}));
}
Console.WriteLine("done_seconds="+timer.Elapsed.TotalSeconds);
