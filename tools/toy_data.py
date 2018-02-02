import numpy as np


class ToyData:
    C = [['a', 'b', 'c'], ['b', 'b'], ['c', 'b', 'a']]
    U = [[.66, .17, .17], [.17, .66, .17], [.17, .17, .66], [.66, .17, .17]]

    def sample_segment(self, utype, cdist):
        segment = self.C[np.random.choice(np.arange(len(self.C)), p=cdist)]
        if self.durations:
            duration = list(np.random.normal(self.dmean[utype], self.dvar[utype], len(segment)))
            return segment, duration
        return segment, None

    def sample_user(self, num_users, udist):
        users = []
        labels = []
        for idx in np.arange(num_users):
            labels.append(np.random.choice(np.arange(len(self.U)), p=udist))
            users.append(self.U[labels[-1]])
        return users, labels

    @classmethod
    def sample(cls, num_users, sdecay=.01, udecay=.1, udist=[.33, .33, .34], durations=False, dmean=[10, 20, 10], dvar=[1.0, 1.0, 1.0]):
        cls.durations = durations
        cls.dmean = dmean
        cls.dvar = dvar
        users = []
        users_durs = []
        cdists, labels = cls.sample_user(cls, num_users, udist)
        for label, cdist in zip(labels, cdists):
            user = []
            user_durs = []
            up = 0.0
            user_active = True
            while user_active:
                session = []
                session_durs = []
                sp = 0.0
                session_active = True
                while session_active:
                    segment, seg_dur = cls.sample_segment(cls, label, cdist)
                    if cls.durations:
                        session_durs = session_durs + seg_dur
                    session = session + segment
                    if np.random.binomial(1, sp):
                        session_active = False
                    sp += sdecay
                user.append(session)
                if cls.durations: user_durs.append(session_durs)
                if np.random.binomial(1, up):
                    user_active = False
                up += udecay
            users.append(user)
            if cls.durations: users_durs.append(user_durs)
        if not cls.durations: users_durs = None
        return users, users_durs, labels
