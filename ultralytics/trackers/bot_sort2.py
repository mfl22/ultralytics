# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path
from collections import deque
import traceback

import numpy as np

from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYWH


# -----------------------------------------------------------------------------

# load feature extractor for re-id
import ultralytics.trackers.feature_extraction as feature_extraction

ENCODER = feature_extraction.TorchReIDFeatureExtractor

# NOTE: this shouldn't be used, it will overwrite previous ( and other camera
# possibly)
DEFAULT_SAVE_DIR = 'runs/track/embeddings'

# -----------------------------------------------------------------------------


class BOTrack(STrack):
    """
    An extended version of the STrack class for YOLOv8,
    adding object tracking features.

    Attributes:
        shared_kalman (KalmanFilterXYWH): A shared Kalman filter for all instances of BOTrack.
        smooth_feat (np.ndarray): Smoothed feature vector.
        curr_feat (np.ndarray): Current feature vector.
        features (deque): A deque to store feature vectors with a maximum length defined by `feat_history`.
        alpha (float): Smoothing factor for the exponential moving average of features.
        mean (np.ndarray): The mean state of the Kalman filter.
        covariance (np.ndarray): The covariance matrix of the Kalman filter.

    Methods:
        update_features(feat): Update features vector and smooth it using exponential moving average.
        predict(): Predicts the mean and covariance using Kalman filter.
        re_activate(new_track, frame_id, new_id): Reactivates a track with updated features and optionally new ID.
        update(new_track, frame_id): Update the YOLOv8 instance with new track and frame ID.
        tlwh: Property that gets the current position in tlwh format `(top left x, top left y, width, height)`.
        multi_predict(stracks): Predicts the mean and covariance of multiple object tracks using shared Kalman filter.
        convert_coords(tlwh): Converts tlwh bounding box coordinates to xywh format.
        tlwh_to_xywh(tlwh): Convert bounding box to xywh format `(center x, center y, width, height)`.

    Usage:
        bo_track = BOTrack(tlwh, score, cls, feat)
        bo_track.predict()
        bo_track.update(new_track, frame_id)
    """

    shared_kalman = KalmanFilterXYWH()

    def __init__(self, tlwh, score, cls, feat=None, feat_history=50):
        """Initialize YOLOv8 object with temporal parameters, such as feature history, alpha and current features."""
        super().__init__(tlwh, score, cls)

        # self.cls initialized in base class
        # self.cls = -1
        self.cls_hist = []  # (cls id, freq)
        self.update_cls(cls, score)

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat):
        """Update features vector and smooth it using exponential moving average."""
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_cls(self, cls, score):
        if len(self.cls_hist) > 0:
            max_freq = 0
            found = False
            for c in self.cls_hist:
                if cls == c[0]:
                    c[1] += score
                    found = True

                if c[1] > max_freq:
                    max_freq = c[1]
                    self.cls = c[0]
            if not found:
                self.cls_hist.append([cls, score])
                self.cls = cls
        else:
            self.cls_hist.append([cls, score])
            self.cls = cls

    def predict(self):
        """Predicts the mean and covariance using Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        ...

    # activate() implemented in base class

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivates a track with updated features and optionally assigns a new ID."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().re_activate(new_track, frame_id, new_id)

    def update(self, new_track, frame_id):
        """Update the YOLOv8 instance with new track and frame ID."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().update(new_track, frame_id)

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y, width, height)`."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def multi_predict(stracks):
        """Predicts the mean and covariance of multiple object tracks using shared Kalman filter."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def convert_coords(self, tlwh):
        """Converts Top-Left-Width-Height bounding box coordinates to X-Y-Width-Height format."""
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width, height)`."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret


class BOTSORT(BYTETracker):
    """
    An extended version of the BYTETracker class for YOLOv8, designed for object tracking with ReID and GMC algorithm.

    Attributes:
        proximity_thresh (float): Threshold for spatial proximity (IoU) between tracks and detections.
        appearance_thresh (float): Threshold for appearance similarity (ReID embeddings) between tracks and detections.
        encoder (object): Object to handle ReID embeddings, set to None if ReID is not enabled.
        gmc (GMC): An instance of the GMC algorithm for data association.
        args (object): Parsed command-line arguments containing tracking parameters.

    Methods:
        get_kalmanfilter(): Returns an instance of KalmanFilterXYWH for object tracking.
        init_track(dets, scores, cls, img): Initialize track with detections, scores, and classes.
        get_dists(tracks, detections): Get distances between tracks and detections using IoU and (optionally) ReID.
        multi_predict(tracks): Predict and track multiple objects with YOLOv8 model.

    Usage:
        bot_sort = BOTSORT(args, frame_rate)
        bot_sort.init_track(dets, scores, cls, img)
        bot_sort.multi_predict(tracks)

    Note:
        The class is designed to work with the YOLOv8 object detection model and supports ReID only if enabled via args.
    """

    def __init__(self, args,
                 encoder=ENCODER,
                 person_class_id=0,frame_rate=30,
                 save_dir=None):
        """Initialize YOLOv8 object with ReID module and GMC algorithm."""
        super().__init__(args, frame_rate)
        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        if args.with_reid:
            # Haven't supported BoT-SORT(reid) yet
            # self.encoder = None

            # encoder/feature extractor has to be provided in with_reid is True
            if encoder is None:
                print('Warning: not using reid ! Encoder required.')
            self.encoder = encoder

            print('bot-sort tracker successfully initialized !\n')
            print(f'using feature extractor: {encoder}')

        self.person_class_id = person_class_id

        # save_dir: first check kwargs, then config file args (args)
        try:
            args_save_dir = args.save_dir
        except Exception:
            traceback.print_exc()
            args_save_dir = None
        self.save_dir = save_dir or args_save_dir

        self.gmc = GMC(method=args.gmc_method)

    def get_kalmanfilter(self):
        """Returns an instance of KalmanFilterXYWH for object tracking."""
        return KalmanFilterXYWH()

    def init_track(self, dets, scores, cls,
                   img=None,
                   features=None,
                   save_features=False, save_fname=None):
        """Initialize track with detections, scores, and classes."""
        if len(dets) == 0:
            return []
        if self.args.with_reid and self.encoder is not None:
            # use person reid only for people :)
            p_ids = np.nonzero(np.array(cls) == self.person_class_id)[0]

            # TODO: only extract needed features (person class id)
            if features is None:
                features_keep = self.encoder.inference(img, dets)
            else:
                features_keep = features
            # save embeddings? don't save here since we will
            # possibly discard some detections...
            # if (
            #     save_features and
            #     (save_fname is not None)
            # ):
            #     np.save(save_fname, features_keep, allow_pickle=True)
            return [
                BOTrack(xyxy, s, c, f) if ind in p_ids
                else BOTrack(xyxy, s, c)
                for ind, (xyxy, s, c, f) in
                enumerate(zip(dets, scores, cls, features_keep))
            ]  # detections
        else:
            return [
                BOTrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)
            ]  # detections

    def get_dists(self, tracks, detections):
        """
        Get distances between tracks and detections using IoU and
        (optionally) ReID embeddings.
        """

        # 1. distances based on IoU
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > self.proximity_thresh

        # only for person detections, use association distance too
        # TODO: is this the good way to handle multiple classes ?
        cls_ids_tracks = np.nonzero(
            [track.cls == self.person_class_id for track in tracks]
        )[0]
        cls_ids = np.nonzero(
            [det.cls == self.person_class_id for det in detections]
        )[0]

        # TODO: mot20
        # if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)

        # 2. embedding distance for subsets of tracks and detections
        # with self.person_class_id

        if self.args.with_reid and self.encoder is not None:
            # extract relevant tracks and detections
            relevant_tracks = [
                track for ind, track in enumerate(tracks)
                if ind in cls_ids_tracks
            ]
            relevant_dets = [
                det for ind, det in enumerate(detections)
                if ind in cls_ids
            ]
            # embedding distances for relevant tracks and detections
            # NOTE: emb-dists shape is:
            # (len(relevant_tracks), len(relevant_dets))
            emb_dists = matching.embedding_distance(
                relevant_tracks, relevant_dets
            ) / 2.0
            # clip large
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            # if IoU already bad, don't even look at embedding distance
            # (NOTE: here we also have to index into relevant tracks and
            # detections)
            emb_dists[dists_mask[cls_ids_tracks][:, cls_ids]] = 1.0
            # update dists only for relevant tracks and detections
            # (update sub-array)
            dists[cls_ids_tracks][:, cls_ids] = np.minimum(
                dists[cls_ids_tracks][:, cls_ids],
                emb_dists
            )
        return dists

    def multi_predict(self, tracks):
        """Predict and track multiple objects with YOLOv8 model."""
        BOTrack.multi_predict(tracks)

    def reset(self):
        """Reset tracker."""
        super().reset()
        self.gmc.reset_params()

    def update(self, results, img, res_obj=None):
        """
        Update tracker with new detections.

        This is overriden from super-class (ByteTracker) update() method.
        """
        # check save_dir; we want to save the embeddings
        if self.save_dir is None:
            if res_obj is not None:
                try:
                    if res_obj.save_dir is None:
                        path = res_obj.path
                        save_dir_name = Path(path).stem
                        self.save_dir = (
                            Path(DEFAULT_SAVE_DIR).joinpath(save_dir_name)
                        )
                    else:
                        save_dir_detector = res_obj.save_dir
                        self.save_dir = Path(save_dir_detector, 'embeddings')
                    self.save_dir.mkdir(mode=511, parents=True,
                                        exist_ok=True)
                except Exception:
                    traceback.print_exc()
                    # self.save_dir = DEFAULT_SAVE_DIR
                    pass

        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = results.conf
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        # Add bbox index
        bboxes = np.concatenate(
            [bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1
        )
        cls = results.cls

        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        inds_second = inds_low & inds_high
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_keep = cls[remain_inds]
        cls_second = cls[inds_second]

        # Extract embeddings
        # NOTE: this is done in init_track()
        # if self.args.with_reid:
        #     features_keep = self.encoder.inference(img, dets)

        if self.save_dir is not None:
            fname = f'embedding_frame_{self.frame_id}.npy'
            save_fname = Path(self.save_dir).joinpath(fname)
        else:
            save_fname = None

        save_only_high = False

        # extract all features
        features = self.encoder.inference(img, bboxes)

        if True:  # not save_only_high:
            if self.frame_id % 1 == 0:
                print(f'\nSaving embeddings to: {save_fname}')
            # save here (all features, even low-confidence)
            # np.save(save_fname, features, allow_pickle=True)
            # subset only high-conf. for init_track()
            features_keep = features[remain_inds]
        else:
            features_keep = None
        detections = self.init_track(dets, scores_keep, cls_keep,
                                     img,
                                     features=features_keep,
                                     save_features=save_only_high,
                                     save_fname=save_fname)

        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # Step 2: First association, with high score detection boxes
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        self.multi_predict(strack_pool)
        if hasattr(self, "gmc") and img is not None:
            warp = self.gmc.apply(img, dets)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        # get dists
        # NOTE: iou and reid included (if reid set)

        dists = self.get_dists(strack_pool, detections)

        # assignment based on (1) iou dist. (motion) and (2) association (reid)

        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.args.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Step 3: Second association, with low score detection boxes association the untrack to the low score detections
        features_second = features[inds_second]
        detections_second = self.init_track(
            dets_second, scores_second, cls_second, img,
            features=features_second,
        )
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # TODO
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        # Deal with unconfirmed tracks, usually tracks with only one beginning frame
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Step 4: Init new stracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # Step 5: Update state
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]  # clip remove stracks to 1000 maximum

        # save features here !
        # get current feature for every track that is left after all above
        # steps
        try:
            features_save = np.asarray(
                [x.curr_feat
                 # if x.curr_feat is not None
                 # should not be necessary but leave this
                 # else self.encoder.inference(img, [x.tlwh])
                 for x in self.tracked_stracks if x.is_activated
                 and x.curr_feat is not None]
            )
        except ValueError:
            print(sum([x.curr_feat is None for x in self.tracked_stracks
                  if x.is_activated]), ' None features. Please check.')
        np.save(save_fname, features_save, allow_pickle=True)

        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)
