# va_player
Audio/media player with voice assistant

# Audio Player with Voice Control

A tool has been implemented as an audio player for playing local music files with voice control. The tool accepts specific requests in XML format and communicates via a REST API with a microservice (backend) to manage the playback of music files.

## Workflow Scenario

*   A helper program first generates a list of music files stored on the disk. During this process, a CSV file is created containing the following information for each track: file path and name, artist (performer, composer), album, track title, and genre.
*   The microservice (backend) is started. It can be accessed via a REST API to control file playback.
*   **For reference:** The `audio.play` tool can execute three different types of requests:

### a) Music Playback Request
Format: `<tool_call><name>audioplay.play_request</name><arguments><artist>ARTIST</artist><title>TITLE</title><genre>GENRE</genre><album>ALBUM</album></arguments></tool_call>`

The arguments within the `<name>` tag are `audioplay.play_request` – the tool name (`audioplay`) and function (request) name `play_request`.

The tool selects the most suitable music from the database (list) based on the request. If several options are found, one is chosen at random. It starts playing the music file corresponding to the request and returns a JSON-formatted response indicating what was selected (fields from the CSV record). If no music is found or an error occurs during startup, this is also reported.
**TODO:** Describe response formats.

### b) Playback Stop Request
Format: `<tool_call><name>audioplay.play_stop</name><arguments></arguments></tool_call>`

Playback is stopped. A message is returned confirming the playback has stopped or reporting any error.

### c) Status Request
Checks whether music is currently playing. If it's impossible to connect to the backend playing the music or another error occurs, this is also reported in the tool's response.

Format: `<tool_call><name>audioplay.play_status</name><arguments></arguments></tool_call>`

## Model Integration (Gemma-3n)

The model (Gemma-3n) receives and recognizes the user's input query. If it recognizes that the query corresponds to one of the three functions listed above (`play_request`, `play_stop`, `play_status`), it generates a text template with one of these functions. If the query does not match, it should return the original query.

**Note:** In practice, it has been observed that the model often does not return the original query but instead starts generating a response as if it were a regular chat conversation with the model.

### Example
If the model recognizes the text "I want to hear Johann Sebastian Bach", it outputs the request text:
`<tool_call><name>audioplay.play_request</name><arguments><artist>Johann Sebastian Bach</artist><title></title><genre></genre><album></album></arguments></tool_call>`
