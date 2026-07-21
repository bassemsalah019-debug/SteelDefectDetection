"""Authentication: signup / login / refresh / logout / me (JWT)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..deps import get_current_user
from ..models import User
from ..schemas import RefreshIn, Token, UserCreate, UserLogin, UserOut
from ..security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])


def _issue(user: User) -> Token:
    return Token(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
        user=UserOut.model_validate(user),
    )


@router.post("/signup", response_model=Token, status_code=status.HTTP_201_CREATED)
def signup(data: UserCreate, db: Session = Depends(get_db)) -> Token:
    email = data.email.lower()
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")
    user = User(email=email, full_name=data.full_name, hashed_password=hash_password(data.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return _issue(user)


@router.post("/login", response_model=Token)
def login(data: UserLogin, db: Session = Depends(get_db)) -> Token:
    user = db.query(User).filter(User.email == data.email.lower()).first()
    if user is None or not verify_password(data.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    return _issue(user)


@router.post("/refresh", response_model=Token)
def refresh(data: RefreshIn, db: Session = Depends(get_db)) -> Token:
    sub = decode_token(data.refresh_token, "refresh")
    if not sub:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired refresh token")
    user = db.get(User, sub)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User no longer exists")
    return _issue(user)


@router.post("/logout")
def logout(_: User = Depends(get_current_user)) -> dict:
    # Stateless JWT: the client discards its tokens. (Token blacklist is post-MVP.)
    return {"status": "ok"}


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> User:
    return user
