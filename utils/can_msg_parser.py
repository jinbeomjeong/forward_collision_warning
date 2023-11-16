import can, cantools, time


class AdasMessage():
    def __init__(self):
        self.__t0 = time.time()
        self.__adas_can_db = cantools.db.load_file(filename='../resource/ADAS_can_protocol.dbc')
        self.__dsm_message = self.__adas_can_db.get_message_by_name(name='DSM')
        self.__fcw_message = self.__adas_can_db.get_message_by_name(name='FCW')
        self.__ldws_message = self.__adas_can_db.get_message_by_name(name='LDWS')

    def __time_elapsed(self) -> float:
        return time.time() - self.__t0

    def create_dsm_can_msg(self, dsm_state: int = 0, timestamp=0.0) -> can.Message():
        data = self.__dsm_message.encode(data={'state_of_DSM': dsm_state})
        message = can.Message(arbitration_id=self.__dsm_message.frame_id, data=data, is_extended_id=False,
                              timestamp=timestamp)
        return message

    def create_fcw_can_msg(self, fcw_state: int = 0, object_type: int = 0, ttc: float = 0.0, timestamp=0.0) -> can.Message():
        data = self.__fcw_message.encode(data={'state_of_FCW': fcw_state, 'object_type': object_type,
                                               'time_to_collision': ttc})
        message = can.Message(arbitration_id=self.__fcw_message.frame_id, data=data, is_extended_id=False,
                              timestamp=timestamp)
        return message

    def create_ldws_can_msg(self, ldws_state: int = 0, timestamp=0.0) -> can.Message():
        data = self.__ldws_message.encode(data={'state_of_LDWS': ldws_state})
        message = can.Message(arbitration_id=self.__ldws_message.frame_id, data=data, is_extended_id=False,
                              timestamp=timestamp)
        return message
