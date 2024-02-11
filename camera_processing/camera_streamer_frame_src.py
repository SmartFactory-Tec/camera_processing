import numpy as np
import asyncio
import json
from typing import AsyncGenerator, Generator
import aiohttp
from aioice import Candidate
from aiortc import RTCPeerConnection, MediaStreamTrack, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.rtcicetransport import candidate_from_aioice
from aiortc.mediastreams import MediaStreamError
from .frame_packet import FramePacket


async def get_track(peer_connection: RTCPeerConnection, logger: any = None) -> MediaStreamTrack:
    track_future: asyncio.Future[MediaStreamTrack] = asyncio.get_running_loop().create_future()

    @peer_connection.on("track")
    def on_track(track: MediaStreamTrack):
        logger.debug("received track")
        track_future.set_result(track)

    await track_future

    return track_future.result()


async def perform_signaling(ws, peer_connection, signaling_future, logger: any = None):
    @peer_connection.on("connectionstatechange")
    def on_connection_state_change():
        logger.debug(peer_connection.connectionState)
        if peer_connection.connectionState == 'connected':
            logger.debug("succesfully connected to webrtc stream")
            signaling_future.set_result(None)

    @peer_connection.on("iceconnectionstatechange")
    async def on_ice_state_change():
        if peer_connection.iceConnectionState == "failed":
            await peer_connection.close()
            signaling_future.exception()

    async for msg in ws:
        parsed_msg = json.loads(msg.data)
        if parsed_msg['type'] == 0 and peer_connection.signalingState == 'stable':
            logger.debug("received remote session description")
            raw_remote_description = parsed_msg['payload']
            remote_description = RTCSessionDescription(raw_remote_description['sdp'],
                                                       raw_remote_description['type'])
            await peer_connection.setRemoteDescription(remote_description)
            answer = await peer_connection.createAnswer()
            await peer_connection.setLocalDescription(answer)
            await ws.send_json({
                'type': 0,
                'payload': {
                    'type': peer_connection.localDescription.type,
                    'sdp': peer_connection.localDescription.sdp,
                },
            })

        elif parsed_msg['type'] == 1:
            if parsed_msg['payload'] is None:
                continue

            logger.debug("received ice candidate")
            ice_candidate_json = parsed_msg['payload']
            candidate_sdp = ice_candidate_json['candidate'].replace('candidate:', '')
            candidate = Candidate.from_sdp(candidate_sdp)
            ice_candidate = candidate_from_aioice(candidate)
            ice_candidate.sdpMid = ice_candidate_json['sdpMid']
            ice_candidate.sdpMLineIndex = ice_candidate_json['sdpMLineIndex']
            await peer_connection.addIceCandidate(ice_candidate)
        else:
            raise RuntimeError("unknown message type received")


async def async_camera_streamer_frame_src(hostname: str, port: int, camera_id: int,
                                          use_https: bool = False, logger: any = None) -> AsyncGenerator[
    FramePacket, any]:
    address = f'http{"s" if use_https else ""}://{hostname}:{port}/{camera_id}'
    session = aiohttp.ClientSession()
    async with session.ws_connect(address) as ws:
        peer_connection = RTCPeerConnection(configuration=RTCConfiguration(
            iceServers=[
                RTCIceServer(urls='stun:stun.l.google.com:19302')
            ]))

        running_loop = asyncio.get_running_loop()
        signaling_future: asyncio.Future = running_loop.create_future()

        track_task = running_loop.create_task(get_track(peer_connection, logger))
        signaling_task = running_loop.create_task(
            perform_signaling(ws, peer_connection, signaling_future, logger))

        track = await track_task
        while True:
            try:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                yield FramePacket(img)
            except MediaStreamError:
                logger.error("unable to get frame from server")
                yield FramePacket(np.zeros((500, 500)))


def camera_streamer_frame_src(hostname: str, port: int, camera_id: int, use_https: bool = False, logger: any = None) \
        -> Generator[FramePacket, None, None]:
    event_loop = asyncio.get_event_loop()
    async_gen = async_camera_streamer_frame_src(hostname, port, camera_id, use_https, logger)
    while True:
        frame_packet = event_loop.run_until_complete(async_gen.__anext__())
        yield frame_packet
