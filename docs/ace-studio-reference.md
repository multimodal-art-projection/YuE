# ACE Studio Capabilities and Workflows

ACE Studio is a music production environment that combines editable singing, AI instruments, audio and MIDI arrangement, hosted plugins, generative audio, and video. Its CLI and MCP connection let an assistant inspect and edit the running project directly. For our music work, it can turn a generated demo into editable parts, synthesize new performances, shape pronunciation and expression, host the installed instruments and effects, and deliver mixes or individual tracks.

Use the capability tables to choose a tool, the workflows to plan a production task, and the technical reference before making edits. This reference is dated November 27, 2026. The installed CLI identifies itself as **0.18.7**, built for **ACE Studio 2.1.9**. External agent access remains experimental, so the installed command schema takes precedence over remembered arguments. [Official connection documentation](https://docs.acestudio.ai/ai-agent/external-agent-access)

## Connection and local setup

Prefer the Windows CLI at `C:/Program Files/ACE Studio/acestudio-cli.exe`. The MCP executable is `C:/Program Files/ACE Studio/ace-mcp-server.exe`. Both connect to the app's local bridge; installing skills alone does not establish that connection. Discover the tool names exposed by the active MCP client before using them.

Install the skills from the [official ACE Studio agent repository](https://github.com/BeatMagic/acestudio_agent_plugin) and locate them in the agent's configured skill directory. Read `ace-studio-setup`, `ace-studio-features`, and `ace-studio-workflows` when their tasks apply. Read `ace-studio-audio-plugins` and a matching recipe before driving a hosted plugin.

In the November 27, 2026 test environment, MCP reported `bridge-ready` and successfully read the project and libraries while the sandboxed CLI could print its manual but reported `not-running`. Verify the available connection in each environment and use a working MCP route when the CLI cannot reach the bridge. A CLI visibility discrepancy does not establish that the app is closed.

For discovery in PowerShell:

```powershell
$aceCli = 'C:/Program Files/ACE Studio/acestudio-cli.exe'
& $aceCli --help
& $aceCli help
& $aceCli help vocalpitch write
& $aceCli help --search 'articulation|phoneme|export'
```

Over MCP, call `get_docs` with exactly one of `path`, `topic`, or `search`. Group tools such as `track_read` and `track_edit` take a `subcommand` and an `arguments` object. Retrieve the selected command's input schema before composing a call. The collected installed manual contains 340 command entries and 18 technical topics, including aliases.

## Observed catalog snapshot

These are dated catalog snapshots, not permanent limits. Instrument plugins are included in both the instrument-source list and plugin registry; the counts should not be added together.

| Resource | Observed availability |
| --- | --- |
| Vocal sources | 165 voices |
| Vocal synthesis models | 6 models, including Verse 2.7 and Verse24 |
| AI instruments | 44 sources |
| External instrument plugins | 40 sources |
| Registered effects | 120 plugins |
| Plugin registry total | 160 entries, comprising 120 effects and 40 instruments |
| Active audio device | AXE IO ASIO at 48 kHz with a 64 sample buffer |

Examples in the installed instrument registry include Scaler 3, Analog Lab V, Lounge Lizard Session 4, SynthMaster Player, Cardinal, and several LANDR tools. A registry entry establishes discoverability; a particular patch, sample library, or license still has to work when the plugin is loaded.

Verse 2.7 currently reports Chinese, Japanese, English, Spanish, Korean, French, Italian, and Portuguese. Query `voice synth-models` for supported languages. The `voice list` language filter instead matches a voice's native training language, which helps select its accent and character.

## Choosing the right capability

| Desired result | Starting point |
| --- | --- |
| Sing a melody with supplied lyrics | Sing clip plus a library voice |
| Change vocal phrasing or pronunciation | Notes, lyrics, phonemes, breaths, and vocal curves |
| Create acoustic instrumental performances | AI Instrument sources and note articulations |
| Use an installed synth or sampled instrument | External instrument slot |
| Generate a complete song | Inspire Me |
| Rework the arrangement or style of existing audio | Music Enhancer |
| Add an accompaniment that follows the current arrangement | Add a Layer |
| Generate ambience, an impact, or a sound texture | Sound Generator |
| Generate narration or dialogue | Speech Generator |
| Separate a mixed recording | Stem Splitter |
| Extract an editable vocal melody and words | Vocal to MIDI |
| Change the timbre of an existing monophonic performance | Voice Changer |
| Deliver a mix, track renders, or an editable timeline | Audio, MIDI, UfData, LRC, video, FCPXML, or AAF export |

## Arrangement and composition

The interface can create, inspect, rename, duplicate, reorder, configure, and delete tracks. Creatable types include Sing, Instrument, generic MIDI, Audio, Video, and Marker. The project already has one chord track; reveal it through the special-track controls rather than attempting to create another. Empty rows are arrangement padding.

Clip tools cover creation of note, marker, and chord clips; splitting, duplication, consolidation, movement, resizing, naming, color, enable state, fades, and audio gain. Media clips come from file import. Video audio can be muted, detached to an audio track, and reattached.

Notes can be added, removed, moved, resized, and split. Sing notes also carry lyrics and language; instrument notes carry articulation. Generic MIDI clips expose velocities, controller lanes, and pitch bend. Chord clips expose chord symbols and durations.

Sing clips enforce monophony: an edit that would leave overlapping notes is rejected with `NOTE_OVERLAP`. Instrument and generic MIDI clips permit overlapping notes, although an AI instrument's musical and physical limits still apply. Clips on one track also have an overlap policy: the default refuses occupied space; `cover` can trim, split, or remove existing material, and video has a separate relocation option.

Transport tools provide play, stop, seek, looping, and metronome control. Separate APIs manage the editing caret, arrangement selection, active editor, visible panels, and special-track rows. Selection is functional state: Music Enhancer, for example, processes the selected time range and track band.

Audio and MIDI recording are also exposed. Set the audio channel or MIDI source through `track set-input`, then move the caret to choose the armed track. Recording is single-target: the caret determines the arm before capture, and the target stays fixed while the take runs. `recording start` reports the armed track and count-in; `recording stop` ends capture and places the take. Input monitoring and audio-device selection have separate controls.

## Vocal synthesis and voice design

Load a library voice onto a track and supply notes and lyrics to create a sung performance. Source discovery includes names, tags, native languages, available model generations, and origins such as premade, cloned, community, and blended voices. Community browsing and collection are exposed. Discover a source's stable reference before loading it, particularly when names collide.

Voice blending combines seeds into a reusable vocal identity. Choir mode places multiple singers on one track, with member gain and mute controls, stereo spread, and timing variation. It is useful for unison texture. Independent harmony lines need separate note parts, normally on separate Sing tracks.

Lyrics have two useful editing levels. Set text directly on individual notes when the syllable division is known, or use `lyric fill` to distribute a sentence through the language-aware engine. A continuation note uses `-` to sustain the preceding syllable across additional pitches. Track language defaults and per-note language are distinct.

Pronunciation tools can derive phonemes from text, list the legal symbols for a language, replace a note's phoneme sequence, shift boundaries, and adjust consonant timing. Reset operations can return pronunciation or timing to model control. Check the phoneme inventory instead of assuming an IPA spelling is accepted.

Custom voice training is a separate cloning workflow in the Voice Library and browser; the current command inventory does not expose a complete training pipeline. [Voice cloning documentation](https://docs.acestudio.ai/voice-cloning/cloning-a-voice-for-vocal-synth)

### Pitch and vibrato

Pitch has four read levels: `ai`, `user`, `combined`, and `final`. They represent the model's contour, the drawn overrides, their combination, and the result after vibrato. Only the drawn user contour is writable. AI contours may still contain gaps before synthesis finishes.

The write payload is a sequence of `[clipLocalTick, absoluteMidiSemitones]` anchors. Fractional pitches are valid. A null value marks a gap; it must not be converted to zero or interpolated across accidentally. The first and last anchors bound the replaced span. Other commands erase a span, copy AI pitch into the drawn curve, scale excursions, change voicing, and add or remove note vibrato.

The final pitch read also reports unvoiced spans. A numeric zero alone cannot establish unvoicedness because zero is also a valid MIDI pitch. Model generations can treat vibrato differently, so inspect the current model and combined contour before treating a flat segment as a manual edit.

### Expression and breaths

Vocal expression is layered. Call `vocalparam layers` first to discover the parameters, writable layers, and data shapes available for the current singer and project. Dynamic shapes delivery intensity; advanced controls include Air, Falsetto, Tension, and Energy. Some singers publish additional style controls. Older projects can retain legacy control routes. [Vocal parameter documentation](https://docs.acestudio.ai/ai-vocal-synth/creating-ai-vocals/vocal-parameters)

Read-only baselines, manual curves, envelopes, and scalar controls are different stores. The engine's effective result is not a writable layer. In the installed interface, writable parameter layers use anchors or scalars; do not send a dense per-tick array merely because a baseline read returned one. Preserve gap markers and use each layer's declared shape.

Breaths are explicit note-associated objects with duration controls. Their audible length can change when leading consonants overlap them. A breath listing therefore reflects the phoneme layout as well as the authored breath duration.

## AI instruments and MIDI plugins

AI instruments turn editable notes into modeled performances. String sections can carry polyphony; most other sources represent a solo player. Write separate lines for winds, and use solo strings' multiple stops sparingly. The preview sample heard during MIDI entry is not the final synthesized performance.

Articulations belong to notes. Read `editor status` for the current `supportedArticulations`; the valid set depends on the loaded instrument and, for an ensemble, on the intersection of its members' capabilities. Smart selection covers many articulations, but explicit pizzicato or mute markings may be necessary. Ensemble mode combines multiple instrument sources with balancing and spread controls.

An external plugin provides a different path: it occupies a track's instrument slot and responds to MIDI, its host parameters, presets, and editor. A note track without a loaded sound source can fall back to piano, so hearing piano is a reason to inspect the source assignment.

MIDI controller lanes use raw MIDI domains rather than plugin normalization: ordinary CC values use 0 to 127, and pitch bend centers at 8192. Read the lane's domain and default before writing. An empty lane means no controller intervention, which differs from a manually drawn flat lane.

## Hosted effects and plugin control

The plugin registry can be filtered by role, vendor, format, name, and category. Windows availability should come from that registry; AU support is a macOS feature. FX APIs can add, remove, reorder, enable, and configure inserts on tracks and the master chain. Prefer an appropriate installed third-party effect when choosing a processor, as the official features skill recommends.

There are three complementary control paths:

1. **Host parameters:** read parameter IDs, types, normalized values, display text, and a fingerprint; write the intended value and verify its display text. Normalization is 0 to 1, and a plugin's mapping to physical units may be nonlinear.
2. **Presets and state:** search or organize preset libraries, save or apply presets, and import or export preset files and whole plugin state. Saved FX chains can be inserted into an existing chain or applied to replace it.
3. **Plugin editor:** open and capture the plugin's own interface, then use scoped clicks, drags, wheel, keys, typing, and resize operations for controls not exposed as parameters.

Prefer ACE Studio's editor channel for hosted plugins. Its captures exclude desktop chrome, and gestures do not move the user's system cursor. Convert captured pixels to logical gesture points using the returned image scale, then capture again to verify the result.

The installed iZotope recipe identifies a specific trap in Ozone, Nectar, and Neutron: parameters can be listed for modules that are absent from the signal chain. Writing one can appear to succeed while having no audible effect. Inspect the editor and add the module before configuring its parameters.

For routine removal from processing, the official skill favors disabling an effect. Bypass serves comparison. Replacing an entire FX chain is a broader edit than inserting one processor, so choose the operation to match the task.

## Generative audio and separation

| Tool | Input and output | Delivery and important behavior |
| --- | --- | --- |
| Inspire Me | An idea, or supplied lyrics plus style, produces a complete song | Staged result; spends song generation credits; cannot cancel server work through `job cancel` |
| Music Enhancer | A selected arrangement region produces a new take or style | Staged result; analysis can supply omitted lyrics and style; spends credits; server work is not cancellable |
| Add a Layer | Other tracks over a named range provide musical context | Direct delivery to a named audio track; the target range is cleared at launch; spends credits |
| Sound Generator | A text prompt, optionally with audio references, produces sound or ambience | Direct delivery; clears the target range; SeedAudio and ElevenLabs have different options; supports a pricing dry run |
| Speech Generator | A script and speakers, or a sequence of dialogue turns, produces speech | Direct delivery at a named position; duration follows the speech; supports a pricing dry run |
| Stem Splitter | An audio clip becomes separated parts | Direct delivery to new tracks; results can finish independently |
| Vocal to MIDI | An audio clip becomes notes and, when requested, lyrics | Direct delivery to a new or specified Sing track; optional source pitch transfer |
| Voice Changer | Rendered audio from a track range becomes another voice or timbre | Direct delivery to new tracks; discover its separate conversion models first |

Inspire Me has idea and lyrics modes. Supplying lyrics selects the latter. Music Enhancer first analyzes its selected source and then produces the new take; style, lyrics, and influence can alter that interpretation.

For Add a Layer, custom mode separates the instrument description from the style prompt. Keep these concise: the reference arrangement already supplies timing and harmony. Use an empty destination track when adding a new part, because generation prepares its destination before the audio exists.

Sound Generator and Speech Generator expose model-specific options. Their dry runs validate and return a current price estimate without preparing a track or opening a job. SeedAudio speech supports up to three declared speakers from library voices, descriptions, or references. ElevenLabs accepts dialogue turns with its own voice and stability controls. Do not mix one provider's arguments into the other's request.

Stem Splitter has four modes. In the installed manual, `basic` separates vocal and accompaniment for free; `professional` separates vocal, drums, bass, piano, guitar, and others for free. `advanced` and prompt-driven `customized` use different credit SKUs. Reverb removal is available only in basic and professional modes. Separation can leave bleed or combine similar instruments.

Vocal to MIDI exposes eight lyric languages plus `notes-only`. Its best input is an isolated monophonic vocal, and the output still needs pitch and rhythm review. Voice Changer preserves an audio performance while changing timbre; it does not create an editable vocal score. Its options include transposition, correction strength, scale correction, variation, and optional accompaniment or reverb removal.

## Background jobs and result retrieval

Generation, separation, scans, tempo analysis, and audio/video export may run as jobs. A successful launch returns a handle; it does not establish that the work finished. Observe lifecycle and individual results with `job get`, `job wait`, and `job results`.

For staged delivery, list the result and explicitly place the chosen asset with `job place`, or download it. For direct delivery, inspect the tracks and clips named by the launch response after the results settle. For computed data such as beat analysis, retrieve the answer through its declared analysis ID. Export jobs produce files, while scan jobs update the registry.

Some staged audio can be played while streaming and later resolves to a final asset. Confirm final completion for delivery. Cancellation support is a property of the job class; removing a result from history is a separate operation from cancelling generation. A multi-result job can leave useful completed parts even when another result fails, so inspect individual outcomes before relaunching anything.

## Timing and rendering

Musical time uses 480 ticks per quarter note. At a constant 120 BPM, one quarter note lasts 0.5 seconds. Use the conversion tools for real projects because tempo and meter can vary across the timeline.

Arrangement positions are global. Note and curve positions are usually local to a clip. Track and collection indices are zero-based, while user-facing track numbers start at one. Human bar positions such as `9.1.0` also differ from wire fields such as zero-based `barPos`. Consult `time-values` and the selected schema rather than applying one numbering rule everywhere.

CLI time flags require explicit forms such as `3840t`, `1.5s`, or a musical position. Bare numbers are refused. Media can be stored natively in seconds while note material is stored in ticks; geometry reads identify `nativeUnit` and provide conversions. Preserve the native values when round-tripping timing.

Beat analysis and applying a beat grid are separate operations. The alpha `tempo apply-beat-analysis` replaces the project's tempo and meter, creates pre-roll to align the first downbeat, and moves existing content. Audio, video, and markers retain timing in seconds under the new grid, with the alignment offset; note-based material retains musical tick relationships. Confirm the intended reference song before applying this to a populated arrangement, and verify the resulting timing.

Vocal Turbo Mode offloads major synthesis work to supported local hardware after a performance test. Pre-rendering starts work after edits when enabled; otherwise playback triggers rendering. The official skills describe AI instruments as cloud-rendered, without the same pre-rendering path. A muted or solo-excluded track may not trigger synthesis. Check source assignment, mute/solo state, and synthesis status before diagnosing a failed render. [Local rendering documentation](https://docs.acestudio.ai/ai-rendering/local-rendering)

## Import export and other DAWs

Import supports common audio formats, MIDI, MusicXML, UfData, video, and still images. A multitrack note file can create one clip per source track. Tempo and meter adoption are explicit options and default off; MIDI/MusicXML can optionally be split into monophonic parts.

| Delivery | Current command surface |
| --- | --- |
| Master mix or individual track renders | WAV or MP3; full content or a specified range |
| WAV settings | 32, 44.1, or 48 kHz; 16, 24, or 32 bit; mono or stereo |
| MP3 settings | 128, 192, 256, or 320 kbps; mono or stereo |
| Editable note data | MIDI for note-bearing tracks; UfData for Sing tracks with lyrics and syllables |
| Timed lyrics | LRC |
| Video render | Picture from the video stack and audio from the master bus; canvas size and frame rate supply defaults |
| Timeline handoff | FCPXML or AAF |
| Library artifacts | Vocal samples and song templates, with membership gates |

MIDI export is synchronous, while audio and video export use jobs. Audio-only performances are absent from a MIDI export. This implementation's MIDI path does not preserve its vocal expression, articulations, lyrics, or syllables; UfData preserves the supported singing text. MusicXML import is available, but MusicXML export is absent from the installed command inventory.

A per-track render bounces authored tracks; stem separation estimates components inside an existing mix. These solve different problems. Verify export scope, effects choices, file completion, and media properties before handing over deliverables.

ACE Bridge 2 connects ACE Studio with another DAW through its instrument plugin, carrying MIDI into linked tracks and audio back. Multiple instances can link different ACE tracks; a track links to one instance at a time. The ARA companion supplies tempo-map and transport synchronization where the host supports it. Host-specific setup remains a separate workflow. [ACE Bridge 2 documentation](https://docs.acestudio.ai/daw-integration/ace-bridge-2)

## Production workflows

### Develop a YuE or Suno demo

1. Import the chosen recording and inspect its placement and beat analysis.
2. Align the project to that recording before detailed editing, after resolving any existing material that would move.
3. Choose the splitter mode needed for the parts being changed; inspect the separated audio.
4. Convert an isolated lead vocal to MIDI and lyrics, then correct transcription errors in musical context.
5. Load the intended singer, refine the words and notes, and shape phonemes, breaths, pitch, vibrato, and delivery.
6. Keep suitable original stems or replace selected parts with AI instruments, plugins, or generated layers.
7. Mix through the installed effects, then export and verify a master and any requested individual tracks.

### Build an original editable arrangement

1. Establish tempo, meter, chords, and sections.
2. Create separate melody and harmony parts, choose voices, and distribute lyrics.
3. Add idiomatic AI instrument parts and plugin instruments; choose articulations from the live supported set.
4. Refine performances and add contextual layers on empty destination tracks where useful.
5. Finish the mix, collect external media into the saved project, and export the requested formats.

### Generate a song and continue editing

Launch Inspire Me from an idea or supplied lyrics, wait for a result, and place the selected take. Align the project to the take, then follow the demo workflow for editable parts. A generated song begins as audio; it does not arrive as a complete editable multitrack score.

### Produce a printable score

Export authored note tracks as MIDI and build the notation in MuseScore or another notation editor. Audio-only parts require transcription first if they are meant to appear. For instruments, the official workflow suggests exporting isolated stems, transcribing locally with Basic Pitch, cleaning the MIDI, then importing it. Use Vocal to MIDI for sung words and melody; do not use a pitch transcriber for drums.

Transcription is an estimate. Remove harmonic duplicates and noise while preserving real ornaments and phrasing. Correct notation durations, rests, enharmonic spelling, staff assignment, and meter. Save the notation source, export PDF and MIDI from that source, reopen it, inspect every rendered score page, and check the exported note data.

## Reliable operation and recovery

All callers share one live project, editor state, selection, and undo stack. Read current tracks, clips, and IDs before acting; prefer UUIDs to indices that can shift. Read project dirty state before switching projects, and use saved copies or the appropriate save operation for work that must be preserved. `collect-save` brings referenced media into the project bundle and returns the actual saved path.

Where supported, carry a read's fingerprint into the matching write using `ifMatch` or CLI `--if-match`. Fingerprints protect specific content scopes. On `STALE_WRITE`, re-read and reconsider the edit; a token from another scope is not interchangeable. Some reads expose change detectors that no write consumes, and merely supplying a token to an unprotected operation does not add protection.

`USER_BUSY` protects an open undo bracket or an export/render that needs a consistent project. A bounded `--wait-busy` waits for that gate; a job's `--wait` waits for completion. They are separate mechanisms. Many ordinary edits create one undo entry, but there is no general multi-command transaction. Undo pops the shared top entry, so inspect history before reversing work while another person is editing.

Use exact command documentation when a summary conflicts with an available operation. For example, the current interface exposes project open/new and history undo/redo even though older narrative passages describe narrower control. Do not blindly replay a write after an uncertain response: first inspect whether its effect or job already exists.

| Symptom | First useful check |
| --- | --- |
| Connected tools but session calls fail | App status and `app awaiting-user`; sign-in or onboarding can require a human |
| Piano instead of the intended sound | Track sound-source assignment |
| Synthesis does not start | Mute/solo state, source readiness, and synthesis status |
| Edit refuses overlap | Read the conflicting notes or clips and choose the intended placement |
| Plugin value changes but sound does not | Verify the active module, bypass/enable state, and editor |
| Generation launch succeeds but no clip appears | Job lifecycle and delivery type; staged assets still require placement |
| Export returns a path but no file is ready | Export job outcome and actual file properties |
| Edit reports stale content | Fresh read and a new fingerprint for the correct scope |

## Product features with separate UI workflows

The product also provides voice training, audio warp editing, a Music Video Director, and related visual production workflows. The current external tool inventory should not be treated as an API for every product feature: it exposes video media editing and export, but no dedicated Director launch or complete voice-training pipeline. Discover the current interface before planning to automate those steps.

Audio Warp stretches the recording to the project grid; applying beat analysis changes the project grid to follow the recording. These are different editing choices. Warp markers and Sync to Project are described in the Audio Editor's UI workflow, while the installed command inventory has no dedicated warp-marker group. [Audio Warp documentation](https://docs.acestudio.ai/project/arrangement-in-canvas/clips/audio-warp)

The Music Video Director handles planning, characters, storyboards, shot generation, and revision through an in-app production process. It has its own model choices and credit costs. Preparing a song and exporting a video through MCP does not establish control over that Director process. [Music Video documentation](https://docs.acestudio.ai/ai-tools/music-video)

## Reference sources and refresh procedure

The operating reference is the installed CLI's `help` corpus and MCP `get_docs`: exact command manuals, input schemas, annotations, and the topics `guardrails`, `interaction-model`, `jobs`, `streaming-results`, `tick-coordinates`, `time-values`, `note-exclusivity`, `phonemes-and-timings`, and `curve-encoding`.

The four installed official skills supply feature choices, production order, and plugin recipes. The upstream [official agent skill repository](https://github.com/BeatMagic/acestudio_agent_plugin) and [ACE Studio documentation index](https://docs.acestudio.ai/llms.txt) provide the broader product reference.

After an ACE Studio update, recheck CLI version, bridge status, exact schemas for the intended task, voice models, source catalogs, plugin availability, and generation pricing. Treat the dated counts and model roster above as discovery aids rather than values to hardcode.
