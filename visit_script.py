import sys
args = sys.argv
args.pop(0) #This is the string of the script itself
city = args.pop(0)  #This is the city
poll = args.pop(0)  #This is the pollutant
min_lon = float(args.pop(0))  #This is the min lon
max_lon = float(args.pop(0))  #This is the max lon
min_lat = float(args.pop(0)) #This is the min lat
max_lat = float(args.pop(0))#This is the max lat
for arg in args:    #The rest of the arguments are strings of dates cooresponding to .3D files
    OpenDatabase("localhost:/Users/jakenoble/DSI/multiplot/data/data_for_visit/{}/{}/{}_{}.3D".format(city.replace(' ','_'),poll,poll,arg), 0)

    ResampleAtts = ResampleAttributes()
    ResampleAtts.useExtents = 1
    ResampleAtts.samplesX = 1000
    ResampleAtts.samplesY = 1000
    ResampleAtts.is3D = 0

    ThresholdAtts = ThresholdAttributes()
    ThresholdAtts.outputMeshType = 0
    ThresholdAtts.listedVarNames = (poll)
    ThresholdAtts.zonePortions = (1)
    ThresholdAtts.lowerBounds = (3)
    ThresholdAtts.upperBounds = (1e+37)
    ThresholdAtts.defaultVarName = poll
    ThresholdAtts.defaultVarIsScalar = 1

    View2DAtts = View2DAttributes()
    View2DAtts.windowCoords = (min_lon, max_lon, min_lat, max_lat)
    View2DAtts.viewportCoords = (0.2, 0.95, 0.15, 0.95)
    View2DAtts.fullFrameActivationMode = View2DAtts.On  # On, Off, Auto
    View2DAtts.fullFrameAutoThreshold = 100
    View2DAtts.xScale = View2DAtts.LINEAR  # LINEAR, LOG
    View2DAtts.yScale = View2DAtts.LINEAR  # LINEAR, LOG
    View2DAtts.windowValid = 0

    SaveWindowAtts = SaveWindowAttributes()
    SaveWindowAtts.outputToCurrentDirectory = 0
    SaveWindowAtts.outputDirectory = "/Users/jakenoble/DSI/multiplot/plots/visIt/{}/{}".format(city.replace(' ','_'),poll)
    SaveWindowAtts.fileName = "{}_{}_".format(poll, arg)

    PseudocolorAtts = PseudocolorAttributes()
    PseudocolorAtts.maxFlag = 1
    PseudocolorAtts.max = 70
    PseudocolorAtts.colorTableName = "hot"
    PseudocolorAtts.invertColorTable = 0
    PseudocolorAtts.opacityType = PseudocolorAtts.Constant  # ColorTable, FullyOpaque, Constant, Ramp, VariableRange
    PseudocolorAtts.opacity = .5

    AddPlot("Pseudocolor", poll, 0, 0)
    SetPlotOptions(PseudocolorAtts)
    SetColorTexturingEnabled(0)
    AddOperator("Project", 0)
    ProjectAtts = ProjectAttributes()
    SetOperatorOptions(ProjectAtts, 0)
    AddOperator("Resample", 0)
    SetOperatorOptions(ResampleAtts, 0)
    AddOperator("Threshold", 0)
    SetOperatorOptions(ThresholdAtts, 0)
    SetView2D(View2DAtts)
    DrawPlots()
    SetSaveWindowAttributes(SaveWindowAtts)
    SaveWindow()
    DeleteActivePlots()
exit()
