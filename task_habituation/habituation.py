#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.1),
    on Mon Apr 15 10:41:32 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.1'
expName = 'habituation'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = False
_loggingLevel = logging.getLevel('warning')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/jjm/sci/memobaby/task_habituation/habituation.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=_loggingLevel)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[600, 400], fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = True
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('start_key') is None:
        # initialise start_key
        start_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='start_key',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    # create speaker 'boing_sound'
    deviceManager.addDevice(
        deviceName='boing_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    if deviceManager.getDevice('key_resp_attention') is None:
        # initialise key_resp_attention
        key_resp_attention = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_attention',
        )
    if deviceManager.getDevice('key_resp_new_stim') is None:
        # initialise key_resp_new_stim
        key_resp_new_stim = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_new_stim',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "start_screen" ---
    task_title = visual.TextBox2(
         win, text='Habituation Task', placeholder='Type here...', font='Arial',
         pos=(0, 0.25),     letterHeight=0.08,
         size=(1, 0.2), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='task_title',
         depth=0, autoLog=True,
    )
    start_instruction = visual.TextBox2(
         win, text='To START press SPACE', placeholder='Type here...', font='Arial',
         pos=(0, 0),     letterHeight=0.04,
         size=(0.8, 0.25), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='start_instruction',
         depth=-1, autoLog=True,
    )
    author_info = visual.TextBox2(
         win, text='Task script by\n    Johannes Julius Mohn\n    Joseph Ventura Kieninger\nBased on a task by\n    Caspar Addyman\nDeveloped in PsychoPy v2024.1.1', placeholder='Type here...', font='Arial',
         pos=(0.8, -0.4),     letterHeight=0.02,
         size=(0.8, 0.25), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='top-left',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='author_info',
         depth=-2, autoLog=True,
    )
    start_key = keyboard.Keyboard(deviceName='start_key')
    
    # --- Initialize components for Routine "trial" ---
    target_stim = visual.ImageStim(
        win=win,
        name='target_stim', 
        image='/Users/jjm/sci/memobaby/task_habituation/stimuli/images/target_transparent_background.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.4, 0.4),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    # Run 'Begin Experiment' code from code_habit_rule
    looktimes = []
    
    # --- Initialize components for Routine "attention_grabber" ---
    moving_dots = visual.DotStim(
        win=win, name='moving_dots',
        nDots=25, dotSize=5000.0,
        speed=0.1, dir=0.0, coherence=1.0,
        fieldPos=(0.0, 0.0), fieldSize=(0.6, 0.6), fieldAnchor='center', fieldShape='circle',
        signalDots='same', noiseDots='walk',dotLife=40.0,
        color=[-1.0,-1.0,0.8], colorSpace='rgb', opacity=None,
        depth=0.0)
    boing_sound = sound.Sound(
        '/Users/jjm/sci/memobaby/task_habituation/stimuli/sounds/boing.mp3', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='boing_sound',    name='boing_sound'
    )
    boing_sound.setVolume(1.0)
    key_resp_attention = keyboard.Keyboard(deviceName='key_resp_attention')
    
    # --- Initialize components for Routine "new_stim" ---
    new_stim_rabbit = visual.ImageStim(
        win=win,
        name='new_stim_rabbit', 
        image='/Users/jjm/sci/memobaby/task_habituation/stimuli/images/rabbit_transparent_background.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.8, 0.6),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_resp_new_stim = keyboard.Keyboard(deviceName='key_resp_new_stim')
    
    # --- Initialize components for Routine "end_screen" ---
    end_text = visual.TextBox2(
         win, text='Das Ende des Experiments wurde erreicht.\n\nVielen Dank für Ihre Teilnahme.', placeholder='Type here...', font='Arial',
         pos=(0, 0),     letterHeight=0.04,
         size=(0.8, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='end_text',
         depth=0, autoLog=True,
    )
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "start_screen" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('start_screen.started', globalClock.getTime(format='float'))
    task_title.reset()
    start_instruction.reset()
    author_info.reset()
    start_key.keys = []
    start_key.rt = []
    _start_key_allKeys = []
    # keep track of which components have finished
    start_screenComponents = [task_title, start_instruction, author_info, start_key]
    for thisComponent in start_screenComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "start_screen" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *task_title* updates
        
        # if task_title is starting this frame...
        if task_title.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            task_title.frameNStart = frameN  # exact frame index
            task_title.tStart = t  # local t and not account for scr refresh
            task_title.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(task_title, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'task_title.started')
            # update status
            task_title.status = STARTED
            task_title.setAutoDraw(True)
        
        # if task_title is active this frame...
        if task_title.status == STARTED:
            # update params
            pass
        
        # *start_instruction* updates
        
        # if start_instruction is starting this frame...
        if start_instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_instruction.frameNStart = frameN  # exact frame index
            start_instruction.tStart = t  # local t and not account for scr refresh
            start_instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_instruction, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_instruction.started')
            # update status
            start_instruction.status = STARTED
            start_instruction.setAutoDraw(True)
        
        # if start_instruction is active this frame...
        if start_instruction.status == STARTED:
            # update params
            pass
        
        # *author_info* updates
        
        # if author_info is starting this frame...
        if author_info.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            author_info.frameNStart = frameN  # exact frame index
            author_info.tStart = t  # local t and not account for scr refresh
            author_info.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(author_info, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'author_info.started')
            # update status
            author_info.status = STARTED
            author_info.setAutoDraw(True)
        
        # if author_info is active this frame...
        if author_info.status == STARTED:
            # update params
            pass
        
        # *start_key* updates
        waitOnFlip = False
        
        # if start_key is starting this frame...
        if start_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_key.frameNStart = frameN  # exact frame index
            start_key.tStart = t  # local t and not account for scr refresh
            start_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_key.started')
            # update status
            start_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(start_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(start_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if start_key.status == STARTED and not waitOnFlip:
            theseKeys = start_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _start_key_allKeys.extend(theseKeys)
            if len(_start_key_allKeys):
                start_key.keys = _start_key_allKeys[-1].name  # just the last key pressed
                start_key.rt = _start_key_allKeys[-1].rt
                start_key.duration = _start_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in start_screenComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "start_screen" ---
    for thisComponent in start_screenComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('start_screen.stopped', globalClock.getTime(format='float'))
    # check responses
    if start_key.keys in ['', [], None]:  # No response was made
        start_key.keys = None
    thisExp.addData('start_key.keys',start_key.keys)
    if start_key.keys != None:  # we had a response
        thisExp.addData('start_key.rt', start_key.rt)
        thisExp.addData('start_key.duration', start_key.duration)
    thisExp.nextEntry()
    # the Routine "start_screen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=12.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial.started', globalClock.getTime(format='float'))
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # Run 'Begin Routine' code from code_habit_rule
        i = trials.thisN
        print(f"Running trial {i}")
        # keep track of which components have finished
        trialComponents = [target_stim, key_resp]
        for thisComponent in trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *target_stim* updates
            
            # if target_stim is starting this frame...
            if target_stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                target_stim.frameNStart = frameN  # exact frame index
                target_stim.tStart = t  # local t and not account for scr refresh
                target_stim.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target_stim, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target_stim.started')
                # update status
                target_stim.status = STARTED
                target_stim.setAutoDraw(True)
            
            # if target_stim is active this frame...
            if target_stim.status == STARTED:
                # update params
                pass
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[0].name  # just the first key pressed
                    key_resp.rt = _key_resp_allKeys[0].rt
                    key_resp.duration = _key_resp_allKeys[0].duration
                    # a response ends the routine
                    continueRoutine = False
            # Run 'Each Frame' code from code_habit_rule
            # get response time and key pressed
            if key_resp.keys:
                rt = key_resp.started - key_resp.rt
                looktimes.append(rt)
                print(f"Registered key press: {_key_resp_allKeys[-1].name}")
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trial.stopped', globalClock.getTime(format='float'))
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        trials.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            trials.addData('key_resp.rt', key_resp.rt)
            trials.addData('key_resp.duration', key_resp.duration)
        # Run 'End Routine' code from code_habit_rule
        print(f"Current response time: {looktimes[i]} s")
        
        # calculate baseline looking time from first three trials
        if i == 3:
            baseline_looktime_key = (looktimes[0] + looktimes[1] + looktimes[2]) / 2
            print(f"baseline_looktime_key is: {baseline_looktime_key} s")
        # compare recent looking times to baseline and determine if experiment is finished
        if i >= 6:
            sum_last_three_looktimes_key = (looktimes[i-2] + looktimes[i-1] + looktimes[i])
            print(f"sum of last three look times is: {sum_last_three_looktimes_key} s")
            if sum_last_three_looktimes_key < baseline_looktime_key:
                trials.finished = True
        # the Routine "trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "attention_grabber" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('attention_grabber.started', globalClock.getTime(format='float'))
        boing_sound.setSound('/Users/jjm/sci/memobaby/task_habituation/stimuli/sounds/boing.mp3', hamming=True)
        boing_sound.setVolume(1.0, log=False)
        boing_sound.seek(0)
        key_resp_attention.keys = []
        key_resp_attention.rt = []
        _key_resp_attention_allKeys = []
        # Run 'Begin Routine' code from loop_boing_sound
        #start sound
        #boing = sound.Sound('/Users/jjm/sci/memobaby/task_habituation/stimuli/sounds/boing.mp3', loops=10, name = 'boing')
        #boing.setVolume(1.0)
        #core.wait(2.0)
        #boing_sound.play(loops=3)
        #boing_soung.close()
        # keep track of which components have finished
        attention_grabberComponents = [moving_dots, boing_sound, key_resp_attention]
        for thisComponent in attention_grabberComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "attention_grabber" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *moving_dots* updates
            
            # if moving_dots is starting this frame...
            if moving_dots.status == NOT_STARTED and tThisFlip >= 2.0-frameTolerance:
                # keep track of start time/frame for later
                moving_dots.frameNStart = frameN  # exact frame index
                moving_dots.tStart = t  # local t and not account for scr refresh
                moving_dots.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(moving_dots, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'moving_dots.started')
                # update status
                moving_dots.status = STARTED
                moving_dots.setAutoDraw(True)
            
            # if moving_dots is active this frame...
            if moving_dots.status == STARTED:
                # update params
                pass
            
            # if boing_sound is starting this frame...
            if boing_sound.status == NOT_STARTED and tThisFlip >= 2.0-frameTolerance:
                # keep track of start time/frame for later
                boing_sound.frameNStart = frameN  # exact frame index
                boing_sound.tStart = t  # local t and not account for scr refresh
                boing_sound.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('boing_sound.started', tThisFlipGlobal)
                # update status
                boing_sound.status = STARTED
                boing_sound.play(when=win)  # sync with win flip
            # update boing_sound status according to whether it's playing
            if boing_sound.isPlaying:
                boing_sound.status = STARTED
            elif boing_sound.isFinished:
                boing_sound.status = FINISHED
            
            # *key_resp_attention* updates
            waitOnFlip = False
            
            # if key_resp_attention is starting this frame...
            if key_resp_attention.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_attention.frameNStart = frameN  # exact frame index
                key_resp_attention.tStart = t  # local t and not account for scr refresh
                key_resp_attention.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_attention, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_attention.started')
                # update status
                key_resp_attention.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_attention.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_attention.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_attention.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_attention.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_attention_allKeys.extend(theseKeys)
                if len(_key_resp_attention_allKeys):
                    key_resp_attention.keys = _key_resp_attention_allKeys[-1].name  # just the last key pressed
                    key_resp_attention.rt = _key_resp_attention_allKeys[-1].rt
                    key_resp_attention.duration = _key_resp_attention_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in attention_grabberComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "attention_grabber" ---
        for thisComponent in attention_grabberComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('attention_grabber.stopped', globalClock.getTime(format='float'))
        boing_sound.pause()  # ensure sound has stopped at end of Routine
        # check responses
        if key_resp_attention.keys in ['', [], None]:  # No response was made
            key_resp_attention.keys = None
        trials.addData('key_resp_attention.keys',key_resp_attention.keys)
        if key_resp_attention.keys != None:  # we had a response
            trials.addData('key_resp_attention.rt', key_resp_attention.rt)
            trials.addData('key_resp_attention.duration', key_resp_attention.duration)
        # the Routine "attention_grabber" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 12.0 repeats of 'trials'
    
    
    # --- Prepare to start Routine "new_stim" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('new_stim.started', globalClock.getTime(format='float'))
    key_resp_new_stim.keys = []
    key_resp_new_stim.rt = []
    _key_resp_new_stim_allKeys = []
    # keep track of which components have finished
    new_stimComponents = [new_stim_rabbit, key_resp_new_stim]
    for thisComponent in new_stimComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "new_stim" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *new_stim_rabbit* updates
        
        # if new_stim_rabbit is starting this frame...
        if new_stim_rabbit.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            new_stim_rabbit.frameNStart = frameN  # exact frame index
            new_stim_rabbit.tStart = t  # local t and not account for scr refresh
            new_stim_rabbit.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(new_stim_rabbit, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'new_stim_rabbit.started')
            # update status
            new_stim_rabbit.status = STARTED
            new_stim_rabbit.setAutoDraw(True)
        
        # if new_stim_rabbit is active this frame...
        if new_stim_rabbit.status == STARTED:
            # update params
            pass
        
        # *key_resp_new_stim* updates
        waitOnFlip = False
        
        # if key_resp_new_stim is starting this frame...
        if key_resp_new_stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_new_stim.frameNStart = frameN  # exact frame index
            key_resp_new_stim.tStart = t  # local t and not account for scr refresh
            key_resp_new_stim.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_new_stim, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_new_stim.started')
            # update status
            key_resp_new_stim.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_new_stim.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_new_stim.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_new_stim.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_new_stim.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_new_stim_allKeys.extend(theseKeys)
            if len(_key_resp_new_stim_allKeys):
                key_resp_new_stim.keys = _key_resp_new_stim_allKeys[-1].name  # just the last key pressed
                key_resp_new_stim.rt = _key_resp_new_stim_allKeys[-1].rt
                key_resp_new_stim.duration = _key_resp_new_stim_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in new_stimComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "new_stim" ---
    for thisComponent in new_stimComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('new_stim.stopped', globalClock.getTime(format='float'))
    # check responses
    if key_resp_new_stim.keys in ['', [], None]:  # No response was made
        key_resp_new_stim.keys = None
    thisExp.addData('key_resp_new_stim.keys',key_resp_new_stim.keys)
    if key_resp_new_stim.keys != None:  # we had a response
        thisExp.addData('key_resp_new_stim.rt', key_resp_new_stim.rt)
        thisExp.addData('key_resp_new_stim.duration', key_resp_new_stim.duration)
    thisExp.nextEntry()
    # the Routine "new_stim" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "end_screen" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('end_screen.started', globalClock.getTime(format='float'))
    end_text.reset()
    # keep track of which components have finished
    end_screenComponents = [end_text]
    for thisComponent in end_screenComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end_screen" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_text* updates
        
        # if end_text is starting this frame...
        if end_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_text.frameNStart = frameN  # exact frame index
            end_text.tStart = t  # local t and not account for scr refresh
            end_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_text.started')
            # update status
            end_text.status = STARTED
            end_text.setAutoDraw(True)
        
        # if end_text is active this frame...
        if end_text.status == STARTED:
            # update params
            pass
        
        # if end_text is stopping this frame...
        if end_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > end_text.tStartRefresh + 5.0-frameTolerance:
                # keep track of stop time/frame for later
                end_text.tStop = t  # not accounting for scr refresh
                end_text.tStopRefresh = tThisFlipGlobal  # on global time
                end_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_text.stopped')
                # update status
                end_text.status = FINISHED
                end_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end_screenComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end_screen" ---
    for thisComponent in end_screenComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('end_screen.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
