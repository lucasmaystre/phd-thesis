from peewee import (
        Model, PrimaryKeyField, TextField, IntegerField, DoubleField,
        DateTimeField, ForeignKeyField, BooleanField, SqliteDatabase, Proxy)
from playhouse.db_url import connect


DB_PROXY = Proxy()


def init_db(db_url):
    database = connect(db_url)
    DB_PROXY.initialize(database)


class BaseModel(Model):
    class Meta:
        database = DB_PROXY


class Team(BaseModel):
    id = PrimaryKeyField()
    swid = IntegerField(unique=True, index=True)
    name = TextField(index=True, null=True)
    country = TextField(null=True)
    alt_names = TextField(null=True)

    def __str__(self):
        return "{:<5} {:<7} | {:<15} | {}".format(
                "Team", self.swid, self.country, self.name)


class Actor(BaseModel):
    id = PrimaryKeyField()
    swid = IntegerField(unique=True, index=True)
    display_name = TextField(index=True, null=True)
    first_name = TextField(null=True)
    last_name = TextField(null=True)
    alt_names = TextField(null=True)
    nationality = TextField(null=True)
    birthdate = DateTimeField(null=True)
    position = TextField(null=True)
    height = DoubleField(null=True)
    weight = DoubleField(null=True)
    foot = TextField(null=True)

    def __str__(self):
        description = "{:<5} {:<7} | {:<15} | ".format(
                "Actor", self.swid, self.nationality)
        description += "{:>17} | ".format(self.display_name)
        description += "{:<31} | ".format(
                str(self.first_name) + " " + str(self.last_name))
        if self.height is not None and self.weight is not None:
            description += "{:>3.0f}cm | {:>2.0f}kg |".format(
                    self.height, self.weight)
        if self.position is not None:
            description += "{:<10}".format(self.position)
        return description


class Game(BaseModel):
    id = PrimaryKeyField()
    swid = IntegerField(unique=True, index=True)
    team_home = ForeignKeyField(
            Team, related_name='games_home', index=True, null=True)
    team_away = ForeignKeyField(
            Team, related_name='games_away', index=True, null=True)
    competition = TextField(null=True)
    phase = TextField(null=True)
    venue = TextField(null=True)
    attendance = IntegerField(null=True)
    kickoff_time = DateTimeField(index=True)
    score_home = IntegerField(null=True)
    score_away = IntegerField(null=True)
    score_home_ht = IntegerField(null=True)
    score_away_ht = IntegerField(null=True)
    score_home_ft = IntegerField(null=True)
    score_away_ft = IntegerField(null=True)
    score_home_et = IntegerField(null=True)
    score_away_et = IntegerField(null=True)
    is_neutral = BooleanField(null=True)
    db_flag = BooleanField(null=True)
    details_url = TextField(null=True)

    def __str__(self):
        description = "{:<5} {:<7} | {:>20} | ".format(
                "Game", self.swid, self.competition)
        description += "{} | ".format(self.kickoff_time)
        description += "{} {} - ".format(self.team_home.name, self.score_home)
        description += "{} {}".format(self.score_away, self.team_away.name)
        return description


class Participant(BaseModel):
    id = PrimaryKeyField()
    game = ForeignKeyField(Game, related_name='participants', index=True)
    team = ForeignKeyField(Team, index=True)
    actor = ForeignKeyField(Actor, related_name='participations', index=True)
    squad_number = IntegerField(null=True)
    is_coach = BooleanField()
    is_starter = BooleanField()
    ratio = DoubleField()


class Odds(BaseModel):
    id = PrimaryKeyField()
    game = ForeignKeyField(Game, related_name='games', index=True)
    platform = TextField()
    type = TextField()
    home_rate = DoubleField(null=True)
    away_rate = DoubleField(null=True)
    draw_rate = DoubleField(null=True)

    def __str__(self):
        description = "'{}' Game id: {} | Platform: {} | ".format(
                "Odds", self.game_id, self.platform)
        description += "Type: {} | ".format(self.type)
        description += "Home odd: {} | Away odd: {} | Draw odd: {} ".format(
                self.home_rate, self.away_rate, self.draw_rate)
        return description


class Prediction(BaseModel):
    id = PrimaryKeyField()
    game = ForeignKeyField(Game, related_name='predictions', index=True)
    probability = DoubleField()
    model = TextField()
    tstamp = DateTimeField()
    details = TextField(null=True)


class Contribution(BaseModel):
    id = PrimaryKeyField()
    weight = DoubleField()
    category = IntegerField()
    actor = ForeignKeyField(Actor, related_name='contribution', index=True)
    model = TextField(null=True)


class CurrentClub(BaseModel):
    id = PrimaryKeyField()
    actor = ForeignKeyField(Actor, related_name="current_club")
    team = ForeignKeyField(Team, null=True)
    squad_number = IntegerField(null=True)
